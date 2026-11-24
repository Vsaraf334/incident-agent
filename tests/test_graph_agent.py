"""Offline tests for the graph's nodes and edge: plain functions on a state dict."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from graph_agent import route, tools_node


def asks_for(*calls: tuple[str, dict]) -> AIMessage:
    """An AIMessage requesting the given tool calls."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"toolu_{i}"}
            for i, (name, args) in enumerate(calls)
        ],
    )


def state_with(last: AIMessage, decisions: dict | None = None) -> dict:
    return {
        "messages": [HumanMessage("alert"), last],
        "calls": [],
        "decisions": decisions or {},
        "rejections": 0,
        "failed_writes": 0,
    }


# ----------------------------------- route ------------------------------------


def test_route_to_approve_when_claude_asks_for_a_tool():
    assert route(state_with(asks_for(("search_logs", {})))) == "approve"


def test_route_to_end_when_claude_answers():
    assert route(state_with(AIMessage(content="The cause is ..."))) == END


# --------------------------------- tools_node ---------------------------------


def test_tools_node_runs_every_call_in_order():
    update = tools_node(
        state_with(
            asks_for(
                ("search_logs", {"service": "auth-service", "query": "429"}),
                ("get_recent_deploys", {"service": "auth-service"}),
            )
        )
    )
    first, second = update["messages"]
    assert isinstance(first, ToolMessage)
    assert (first.tool_call_id, second.tool_call_id) == ("toolu_0", "toolu_1")
    assert "Too Many Requests" in first.content
    assert second.content.startswith("v1.8.0")
    assert first.status == second.status == "success"


def test_tools_node_records_the_trace():
    update = tools_node(
        state_with(
            asks_for(("search_logs", {"service": "auth-service", "query": "429"}))
        )
    )
    [call] = update["calls"]
    assert call.name == "search_logs"
    assert call.input == {"service": "auth-service", "query": "429"}
    assert call.is_error is False


def test_tools_node_turns_exceptions_into_error_results():
    update = tools_node(
        state_with(
            asks_for(
                ("search_logs", {"service": "checkout", "query": "ERROR"}),
                ("restart_service", {}),
            )
        )
    )
    bad_service, unknown_tool = update["messages"]
    assert bad_service.status == unknown_tool.status == "error"
    assert "Known services:" in bad_service.content
    assert [c.is_error for c in update["calls"]] == [True, True]


# ------------------- approval, end to end with a fake Claude -------------------
# These run the real graph (interrupts, checkpointer, resume) with a scripted
# "Claude", stub approvers and a fake file_incident. No API calls.

import pytest

import agent
import graph_agent

PROPOSAL = {"title": "T", "cause": "C", "evidence": "E", "severity": "SEV2"}
ANSWER = AIMessage(
    content="Summary: ...", response_metadata={"stop_reason": "end_turn"}
)


def reply(*calls: tuple[str, dict]) -> AIMessage:
    message = asks_for(*calls)
    message.response_metadata = {"stop_reason": "tool_use"}
    return message


class FakeClaude:
    """Returns the scripted replies in order, whatever it is asked."""

    def __init__(self, *replies: AIMessage):
        self.replies = list(replies)

    def invoke(self, _messages):
        return self.replies.pop(0)


@pytest.fixture
def filed(monkeypatch):
    incidents = []

    def fake_file_incident(**kwargs):
        incidents.append(kwargs)
        return f"Filed INC-00{len(incidents)}"

    monkeypatch.setitem(agent.FUNCTIONS, "file_incident", fake_file_incident)
    return incidents


def run_with(monkeypatch, approve, *replies: AIMessage):
    fake = FakeClaude(*replies)
    monkeypatch.setattr(graph_agent, "make_llm", lambda: fake)
    return graph_agent.run_graph("alert", approve=approve, verbose=False)


class Approver:
    """A stub approver that answers from a list and counts how often it was asked."""

    def __init__(self, *answers: tuple[bool, str]):
        self.answers = list(answers)
        self.asked = 0

    def __call__(self, _block):
        self.asked += 1
        return self.answers.pop(0)


def test_read_tools_never_ask(monkeypatch, filed):
    approver = Approver()
    run = run_with(
        monkeypatch,
        approver,
        reply(("search_logs", {"service": "auth-service", "query": "429"})),
        ANSWER,
    )
    assert approver.asked == 0
    assert run.calls[0].approved is None
    assert run.answer == "Summary: ..."


def test_approved_incident_is_filed(monkeypatch, filed):
    run = run_with(
        monkeypatch, Approver((True, "")), reply(("file_incident", PROPOSAL)), ANSWER
    )
    assert filed == [PROPOSAL]
    assert run.calls[0].approved is True


def test_rejected_incident_is_not_filed(monkeypatch, filed):
    run = run_with(
        monkeypatch, Approver((False, "")), reply(("file_incident", PROPOSAL)), ANSWER
    )
    assert filed == []
    assert run.calls[0].approved is False
    assert "Do not file it again" in run.calls[0].output
    assert run.rejections == 1


def test_two_writes_in_one_reply_are_each_filed_once(monkeypatch, filed):
    # Resuming re-runs approve_node from the top. The writes happen in tools_node,
    # after all the questions, so nothing is filed twice.
    approver = Approver((True, ""), (True, ""))
    run_with(
        monkeypatch,
        approver,
        reply(
            ("file_incident", PROPOSAL), ("file_incident", dict(PROPOSAL, title="T2"))
        ),
        ANSWER,
    )
    assert approver.asked == 2
    assert [i["title"] for i in filed] == ["T", "T2"]


def test_stops_asking_after_max_rejections(monkeypatch, filed):
    approver = Approver((False, ""), (False, ""))
    run = run_with(
        monkeypatch,
        approver,
        reply(("file_incident", PROPOSAL)),
        reply(("file_incident", PROPOSAL)),
        reply(("file_incident", PROPOSAL)),
        ANSWER,
    )
    assert approver.asked == agent.MAX_REJECTIONS
    assert "Do not call file_incident again" in run.calls[-1].output
    assert filed == []
