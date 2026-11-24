"""MCP server for incident-agent: lets an MCP host (e.g. Claude Code) investigate alerts.

A host starts this file as a subprocess and talks to it over stdin/stdout, one
JSON message per line. Never print() in this file: stdout is the protocol channel.

Over MCP, the HOST owns the agent loop and the approval step, not us. What this
server can still guarantee or request:
- the content checks in tools.py run on every call (our code, any host);
- file_incident asks Claude Code to show a permission prompt on EVERY call
  (_meta "anthropic/requiresUserInteraction"): allow rules and bypass mode can't
  skip it, and where nobody can answer, the call is denied. Other hosts ignore it.
"""

from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

import tools
from agent import SYSTEM_PROMPT
from tools import MAX_DEPLOYS, MAX_LOG_LINES

# Where file_incident writes. A module variable so the tests can point it at tmp_path.
INCIDENTS_DIR = tools.INCIDENTS_DIR

server = MCPServer(
    "incident-agent",
    # The same on-call rules as the loop's system prompt. The host decides whether
    # to use them (week 9: the server offers, the host decides).
    instructions=SYSTEM_PROMPT,
)

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
Service = Annotated[str, Field(description="Service name, e.g. checkout-service")]


@server.tool(annotations=READ_ONLY)
def get_recent_deploys(
    service: Service,
    limit: Annotated[
        int, Field(description="How many deploys", ge=1, le=MAX_DEPLOYS)
    ] = 3,
) -> str:
    """The service's most recent deploys, newest first: version, time (UTC), author and change."""
    try:
        return tools.get_recent_deploys(service, limit)
    except ValueError as e:
        # ToolError, so the model sees "Unknown service ... Known services: ...".
        # Any other exception would show only "Error executing tool" (week 9).
        raise ToolError(str(e)) from e


@server.tool(annotations=READ_ONLY)
def search_logs(
    service: Service,
    query: Annotated[
        str, Field(description="Text to find, e.g. ERROR, Timeout, or a class name")
    ],
    limit: Annotated[
        int, Field(description="How many lines at most", ge=1, le=MAX_LOG_LINES)
    ] = 20,
) -> str:
    """Log lines of the service that contain the query (case-insensitive), oldest first."""
    try:
        return tools.search_logs(service, query, limit)
    except ValueError as e:
        raise ToolError(str(e)) from e


@server.tool(
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,  # it adds a record; it never deletes or overwrites one
        idempotent_hint=False,  # calling it twice files two incidents
        open_world_hint=False,
    ),
    # Claude Code: show a permission prompt on every call, whatever the user's settings
    meta={"anthropic/requiresUserInteraction": True},
)
def file_incident(
    title: Annotated[str, Field(description="One line, e.g. 'Checkout 502s'")],
    cause: Annotated[str, Field(description="The root cause, in one sentence")],
    evidence: Annotated[
        str,
        Field(description="The log lines and deploys that support the cause, quoted"),
    ],
    severity: Annotated[
        Literal["SEV1", "SEV2", "SEV3"],
        Field(
            description="SEV1: outage. SEV2: a main feature failing for many users. SEV3: degraded or minor."
        ),
    ],
) -> str:
    """File an incident report. Side effects: creates a record and notifies the team. Only call it once you know the cause."""
    try:
        return tools.file_incident(
            title, cause, evidence, severity, incidents_dir=INCIDENTS_DIR
        )
    except ValueError as e:
        # e.g. leaked tool-call markup: the check ran before writing, so nothing was filed
        raise ToolError(str(e)) from e


if __name__ == "__main__":
    server.run()
