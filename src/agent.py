"""The incident agent: Claude investigates an alert with the three tools in tools.py.

Run:  uv run --env-file .env python src/agent.py "checkout-service: 5xx error rate at 12%"

Step 2: file_incident runs WITHOUT asking anyone. Approval comes in step 3.
"""

import sys
from dataclasses import dataclass, field

import anthropic
from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam, ToolUseBlock

from tools import (
    MAX_DEPLOYS,
    MAX_LOG_LINES,
    SEVERITIES,
    file_incident,
    get_recent_deploys,
    search_logs,
)

client = anthropic.Anthropic()
MODEL = "claude-haiku-5-5"
MAX_STEPS = 8  # API calls; an investigation needs more than a docs search did

SYSTEM_PROMPT = """You are the on-call engineer for an online shop. You get an alert and \
investigate it with the tools.

- Look at the evidence before you conclude anything: the service's logs and its recent deploys.
- Only blame a deploy if the errors started after it. Compare the timestamps.
- The cause can be outside the service: a database, another service, an external provider.
- File an incident only when the evidence points to a clear cause. Quote the log lines and \
deploys that support it.
- If the evidence is weak or there are no errors, file nothing. Say what you checked and \
what you found.

End with a short summary: the cause (or "unknown"), the evidence, and whether you filed \
an incident."""

# tool name -> function (a Map<String, Handler>)
FUNCTIONS = {
    "get_recent_deploys": get_recent_deploys,
    "search_logs": search_logs,
    "file_incident": file_incident,
}

SERVICE = {"type": "string", "description": "Service name, e.g. checkout-service"}

# Every tool is strict: the API guarantees Claude's input matches the schema
# (all required fields present, no extra ones, enums respected). Strict schemas
# need "additionalProperties": False and can't use minimum/maximum, so the limit
# ranges are in the descriptions and enforced in tools.py.
TOOLS: list[ToolParam] = [
    {
        "name": "get_recent_deploys",
        "description": "The service's most recent deploys, newest first: version, "
        "time (UTC), author and what changed.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "service": SERVICE,
                "limit": {
                    "type": "integer",
                    "description": f"How many deploys, 1 to {MAX_DEPLOYS}",
                    "default": 3,
                },
            },
            "required": ["service"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_logs",
        "description": "Log lines of the service that contain the query "
        "(case-insensitive), oldest first. Each line starts with its UTC timestamp.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "service": SERVICE,
                "query": {
                    "type": "string",
                    "description": "Text to find, e.g. ERROR, Timeout, or a class name",
                },
                "limit": {
                    "type": "integer",
                    "description": f"How many lines at most, 1 to {MAX_LOG_LINES}",
                    "default": 20,
                },
            },
            "required": ["service", "query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "file_incident",
        "description": "File an incident report. This has side effects: it creates a "
        "record and notifies the team. Only call it once you know the cause.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title": {
                    "type": "string",
                    "description": "One line, e.g. 'Checkout 502s'",
                },
                "cause": {
                    "type": "string",
                    "description": "The root cause, in one sentence",
                },
                "evidence": {
                    "type": "string",
                    "description": "The log lines and deploys that support the cause, quoted",
                },
                "severity": {
                    "type": "string",
                    "enum": list(SEVERITIES),
                    "description": "SEV1: outage. SEV2: a main feature failing for many "
                    "users. SEV3: degraded or minor.",
                },
            },
            "required": ["title", "cause", "evidence", "severity"],
        },
    },
]


@dataclass
class ToolCall:
    """One tool call, as the evals will see it."""

    name: str
    input: dict
    output: str
    is_error: bool


@dataclass
class AgentRun:
    """Everything about one run: the answer and the path the agent took."""

    alert: str
    answer: str = ""
    calls: list[ToolCall] = field(default_factory=list)  # a NEW list per run
    steps: int = 0  # API calls
    input_tokens: int = 0
    output_tokens: int = 0


def run_tool(block: ToolUseBlock, run: AgentRun) -> ToolResultBlockParam:
    """Run one tool_use block, record it in run.calls, and return its tool_result.

    Never raises: any exception becomes a tool_result with is_error=True,
    whose content is the exception message (Claude reads it and can retry).
    """
    try:
        output = FUNCTIONS[block.name](**block.input)
        is_error = False
    except Exception as e:  # noqa: BLE001 - any failure goes back to Claude as is_error
        output = str(e)
        is_error = True

    run.calls.append(
        ToolCall(name=block.name, input=block.input, output=output, is_error=is_error)
    )
    result: ToolResultBlockParam = {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": output,
        "is_error": is_error,
    }
    return result


def run_agent(alert: str) -> AgentRun:
    """Loop until Claude stops asking for tools. Returns the run, with its trace."""
    run = AgentRun(alert=alert)
    messages: list[MessageParam] = [{"role": "user", "content": alert}]

    for step in range(1, MAX_STEPS + 1):
        response = client.messages.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
        )

        run.steps += 1
        run.input_tokens += response.usage.input_tokens
        run.output_tokens += response.usage.output_tokens

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            run.answer = "".join(b.text for b in response.content if b.type == "text")
            if not run.answer.strip():  # the on-call engineer must always get a report
                raise RuntimeError("Agent ended without an answer")
            return run
        if response.stop_reason != "tool_use":
            raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason}")

        calls = [b for b in response.content if b.type == "tool_use"]
        for b in calls:
            print(f"  step {step}: {b.name}({b.input})")

        for b in response.content:
            if b.type == "text" and b.text.strip():
                print(f"  step {step} thinks: {b.text.strip()}")

        results = [run_tool(b, run) for b in calls]
        messages.append({"role": "user", "content": results})

    raise RuntimeError(f"Agent did not finish within {MAX_STEPS} steps")


# ================================ TRY IT ======================================

if __name__ == "__main__":
    alert = " ".join(sys.argv[1:]) or "checkout-service: 5xx error rate at 12%"
    run = run_agent(alert)
    print(f"\n{run.answer}\n")
    print(
        f"{run.steps} API calls, {len(run.calls)} tool calls, "
        f"{run.input_tokens} input + {run.output_tokens} output tokens"
    )
