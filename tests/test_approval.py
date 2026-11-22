"""Offline tests for handle_call: the approval rules. No API calls, no console input."""

import pytest
from anthropic.types import ToolUseBlock

import agent
from agent import MAX_REJECTIONS, AgentRun, handle_call

PROPOSAL = {"title": "T", "cause": "C", "evidence": "E", "severity": "SEV2"}


def block(name: str, **tool_input) -> ToolUseBlock:
    return ToolUseBlock(id="toolu_1", name=name, input=tool_input, type="tool_use")


def say_yes(_block):
    return True, ""


def say_no(_block):
    return False, ""


def never_ask(_block):
    raise AssertionError("the approver should not have been asked")


@pytest.fixture
def filed(monkeypatch):
    """Replace file_incident with a fake that records what would have been filed."""
    incidents = []

    def fake_file_incident(**kwargs):
        incidents.append(kwargs)
        return "Filed INC-001: T"

    monkeypatch.setitem(agent.FUNCTIONS, "file_incident", fake_file_incident)
    return incidents


def test_read_tools_run_without_asking(filed):
    run = AgentRun(alert="a")
    result = handle_call(
        block("search_logs", service="auth-service", query="429"), run, never_ask
    )
    assert "Too Many Requests" in result["content"]
    assert run.calls[0].approved is None


def test_approved_incident_is_filed(filed):
    run = AgentRun(alert="a")
    result = handle_call(block("file_incident", **PROPOSAL), run, say_yes)
    assert result["content"] == "Filed INC-001: T"
    assert result.get("is_error") is not True
    assert filed == [PROPOSAL]
    assert run.calls[0].approved is True


def test_rejected_incident_is_not_filed(filed):
    run = AgentRun(alert="a")
    result = handle_call(block("file_incident", **PROPOSAL), run, say_no)
    assert filed == []
    assert result["is_error"] is True
    assert "Do not file it again" in result["content"]
    assert run.calls[0].approved is False
    assert run.rejections == 1


def test_rejection_reason_reaches_claude(filed):
    run = AgentRun(alert="a")
    reason = "payment-service is the cause, check its deploys"
    result = handle_call(
        block("file_incident", **PROPOSAL), run, lambda _b: (False, reason)
    )
    assert reason in result["content"]
    assert "Do not file it again" not in result["content"]


def test_stops_asking_after_max_rejections(filed):
    run = AgentRun(alert="a")
    for _ in range(MAX_REJECTIONS):
        handle_call(block("file_incident", **PROPOSAL), run, say_no)
    result = handle_call(block("file_incident", **PROPOSAL), run, never_ask)
    assert result["is_error"] is True
    assert "Do not call file_incident again" in result["content"]
    assert filed == []
    assert len(run.calls) == MAX_REJECTIONS + 1


def test_tool_error_after_approval_is_still_an_error():
    # Approved, but the real file_incident rejects the garbled content: nothing filed.
    run = AgentRun(alert="a")
    garbled = dict(PROPOSAL, cause='C</cause>\n<parameter name="evidence">E')
    result = handle_call(block("file_incident", **garbled), run, say_yes)
    assert result["is_error"] is True
    assert "tool-call markup" in result["content"]
