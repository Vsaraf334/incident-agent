"""Offline tests for run_tool: no API calls. The loop itself is tested by the evals."""

from anthropic.types import ToolUseBlock

from agent import FUNCTIONS, TOOLS, AgentRun, run_tool


def block(name: str, **tool_input) -> ToolUseBlock:
    return ToolUseBlock(id="toolu_1", name=name, input=tool_input, type="tool_use")


def test_every_tool_definition_has_a_function():
    assert [t["name"] for t in TOOLS] == list(FUNCTIONS)


def test_success_result():
    run = AgentRun(alert="a")
    result = run_tool(
        block("get_recent_deploys", service="payment-service", limit=1), run
    )
    assert result["tool_use_id"] == "toolu_1"
    assert result["content"].startswith("v4.1.1")
    assert result.get("is_error") is not True


def test_success_is_recorded():
    run = AgentRun(alert="a")
    run_tool(block("search_logs", service="auth-service", query="429"), run)
    [call] = run.calls
    assert call.name == "search_logs"
    assert call.input == {"service": "auth-service", "query": "429"}
    assert "Too Many Requests" in call.output
    assert call.is_error is False


def test_tool_error_becomes_is_error_result():
    run = AgentRun(alert="a")
    result = run_tool(block("search_logs", service="checkout", query="ERROR"), run)
    assert result["is_error"] is True
    assert "Known services:" in result["content"]
    assert run.calls[0].is_error is True


def test_unknown_tool_does_not_crash():
    run = AgentRun(alert="a")
    result = run_tool(block("restart_service", service="checkout-service"), run)
    assert result["is_error"] is True


def test_missing_argument_does_not_crash():
    run = AgentRun(alert="a")
    result = run_tool(block("search_logs", service="auth-service"), run)
    assert result["is_error"] is True


def test_each_run_has_its_own_calls():
    first, second = AgentRun(alert="a"), AgentRun(alert="b")
    run_tool(block("search_logs", service="auth-service", query="429"), first)
    assert second.calls == []


def test_every_tool_is_strict():
    for tool in TOOLS:
        assert tool.get("strict") is True
        assert tool["input_schema"].get("additionalProperties") is False


def test_agent_error_keeps_the_trace(monkeypatch):
    """A failed run raises AgentError, and e.run still holds what happened."""
    import pytest
    from anthropic.types import Message, Usage

    import agent

    def fake_create(**_kwargs):
        return Message(
            id="msg_1",
            type="message",
            role="assistant",
            model="fake",
            content=[],
            stop_reason="max_tokens",
            usage=Usage(input_tokens=100, output_tokens=4096),
        )

    monkeypatch.setattr(agent.client.messages, "create", fake_create)
    with pytest.raises(agent.AgentError, match="max_tokens") as info:
        agent.run_agent("checkout-service: 5xx", verbose=False)
    assert info.value.run.steps == 1
    assert info.value.run.output_tokens == 4096
