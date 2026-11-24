"""Tests for the MCP server, through a real MCP client connected in-process.

Client(server) talks to the server object directly (no subprocess, no stdio), but
every call still goes through MCP: tool listing, argument validation, errors.
"""

import asyncio

import pytest
from mcp import Client
from mcp.types import CallToolResult, TextContent

import mcp_server

INCIDENT = {"title": "T", "cause": "C", "evidence": "E", "severity": "SEV2"}


def text_of(result: CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def run(coro):
    return asyncio.run(coro)


async def list_tools():
    async with Client(mcp_server.server) as client:
        return {t.name: t for t in (await client.list_tools()).tools}


async def call(tool: str, arguments: dict) -> CallToolResult:
    async with Client(mcp_server.server) as client:
        return await client.call_tool(tool, arguments)


@pytest.fixture
def incidents(tmp_path, monkeypatch):
    """file_incident writes to a temporary folder, never to incidents/."""
    monkeypatch.setattr(mcp_server, "INCIDENTS_DIR", tmp_path)
    return tmp_path


# ------------------------------ what hosts see ---------------------------------


def test_lists_the_three_tools():
    assert sorted(run(list_tools())) == [
        "file_incident",
        "get_recent_deploys",
        "search_logs",
    ]


def test_only_file_incident_requires_user_interaction():
    tools = run(list_tools())
    assert tools["file_incident"].meta == {"anthropic/requiresUserInteraction": True}
    assert tools["search_logs"].meta is None
    assert tools["get_recent_deploys"].meta is None


def test_read_tools_are_marked_read_only():
    tools = run(list_tools())
    for name in ("search_logs", "get_recent_deploys"):
        assert tools[name].annotations.read_only_hint is True
    assert tools["file_incident"].annotations.read_only_hint is False


def test_severity_is_an_enum_and_limits_are_bounded():
    tools = run(list_tools())
    severity = tools["file_incident"].input_schema["properties"]["severity"]
    assert severity["enum"] == ["SEV1", "SEV2", "SEV3"]
    limit = tools["search_logs"].input_schema["properties"]["limit"]
    assert (limit["minimum"], limit["maximum"]) == (1, 50)


def test_instructions_carry_the_on_call_rules():
    async def instructions():
        async with Client(mcp_server.server) as client:
            return client.instructions

    assert "Only blame a deploy if the errors started after it" in run(instructions())


# --------------------------------- calls ---------------------------------------


def test_search_logs_returns_the_lines():
    result = run(call("search_logs", {"service": "auth-service", "query": "429"}))
    assert result.is_error is False
    assert "Too Many Requests" in text_of(result)


def test_tool_errors_reach_the_model():
    result = run(call("get_recent_deploys", {"service": "checkout"}))
    assert result.is_error is True
    assert "Known services: auth-service" in text_of(result)


def test_out_of_range_limit_is_rejected():
    result = run(
        call("search_logs", {"service": "auth-service", "query": "x", "limit": 500})
    )
    assert result.is_error is True


def test_file_incident_writes_the_incident(incidents):
    result = run(call("file_incident", INCIDENT))
    assert text_of(result) == "Filed INC-001: T"
    assert (incidents / "INC-001.md").exists()


def test_garbled_incident_is_rejected_and_nothing_is_written(incidents):
    garbled = dict(INCIDENT, cause='C</cause>\n<parameter name="evidence">E')
    result = run(call("file_incident", garbled))
    assert result.is_error is True
    assert "tool-call markup" in text_of(result)
    assert list(incidents.iterdir()) == []


def test_invalid_severity_is_rejected_before_the_tool_runs(incidents):
    result = run(call("file_incident", dict(INCIDENT, severity="high")))
    assert result.is_error is True
    assert list(incidents.iterdir()) == []
