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


def state_with(last: AIMessage) -> dict:
    return {"messages": [HumanMessage("alert"), last], "calls": []}


# ----------------------------------- route ------------------------------------


def test_route_to_tools_when_claude_asks_for_one():
    assert route(state_with(asks_for(("search_logs", {})))) == "tools"


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
