"""The incident agent: Claude investigates an alert with the three tools in tools.py.

Run:  uv run --env-file .env python src/agent.py "checkout-service: 5xx error rate at 12%"

Read tools run freely. file_incident (a write) runs only after a human approves it.
"""

import sys
import textwrap
from collections.abc import Callable
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
MODEL = "claude-sonnet-5-5"  # Haiku: same causes, but 7 of 45 eval runs could not file (README)
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

- If the engineer rejects an incident, follow their reason if they give one. Without a \
reason, don't file it again.

End with a short summary: the cause (or "unknown"), the evidence, and whether you filed \
an incident."""

# tool name -> function (a Map<String, Handler>)
FUNCTIONS = {
    "get_recent_deploys": get_recent_deploys,
    "search_logs": search_logs,
    "file_incident": file_incident,
}

NEEDS_APPROVAL = {"file_incident"}  # write tools: a human says yes first
MAX_REJECTIONS = 2  # after this many "no"s, stop asking and reject automatically
MAX_FAILED_WRITES = 2  # after this many approved-but-failed writes, stop trying

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
    approved: bool | None = None  # None: no approval needed (a read tool)


@dataclass
class AgentRun:
    """Everything about one run: the answer and the path the agent took."""

    alert: str
    answer: str = ""
    calls: list[ToolCall] = field(default_factory=list)  # a NEW list per run
    steps: int = 0  # API calls
    input_tokens: int = 0
    output_tokens: int = 0
    rejections: int = 0
    failed_writes: int = 0  # approved write calls that the tool itself rejected


class AgentError(RuntimeError):
    """The agent failed. .run holds everything it did up to the failure.

    Like a Java exception that carries context: the caller (the evals) can still
    see the trace, instead of only the message.
    """

    def __init__(self, message: str, run: AgentRun) -> None:
        super().__init__(message)
        self.run = run


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


# An approver gets Claude's proposed call and returns (approved, reason).
# Production asks a person; the evals and tests pass in an automatic one.
Approver = Callable[[ToolUseBlock], tuple[bool, str]]


def ask_human(block: ToolUseBlock) -> tuple[bool, str]:
    """Show the proposed incident and ask on the console. Enter alone means no."""
    proposal = block.input
    print("\n  Claude wants to file an incident:")
    print(f"    title:    {proposal['title']}")
    print(f"    severity: {proposal['severity']}")
    print(f"    cause:    {proposal['cause']}")
    print(f"    evidence:\n{textwrap.indent(str(proposal['evidence']), '      ')}")
    if input("  File it? [y/N] ").strip().lower() == "y":
        return True, ""
    return False, input("  Reason (optional, Enter to skip): ").strip()


def rejection_message(reason: str) -> str:
    """What Claude reads when the engineer says no."""
    if reason:
        return f"Rejected by the on-call engineer. Nothing was filed. Their reason: {reason}"
    return (
        "Rejected by the on-call engineer. Nothing was filed. "
        "Do not file it again; report your findings instead."
    )


TOO_MANY_REJECTIONS = (
    f"Not filed: the engineer has rejected {MAX_REJECTIONS} incidents in this "
    "investigation. Do not call file_incident again; report your findings instead."
)


TOO_MANY_FAILED_WRITES = (
    f"Not filed: file_incident has already failed {MAX_FAILED_WRITES} times in this "
    "investigation. Do not call it again; report your findings, and say that "
    "the incident still needs to be filed."
)


def reject(block: ToolUseBlock, run: AgentRun, message: str) -> ToolResultBlockParam:
    """Record a rejected call (the tool never ran) and return its error result."""
    run.calls.append(
        ToolCall(block.name, block.input, message, is_error=True, approved=False)
    )
    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": message,
        "is_error": True,
    }


# ================================= YOUR PART ==================================


def handle_call(
    block: ToolUseBlock, run: AgentRun, approve: Approver
) -> ToolResultBlockParam:
    """Run one tool call, asking for approval first if the tool needs it.

    - a tool not in NEEDS_APPROVAL: just run_tool(block, run)
    - already MAX_REJECTIONS rejections in this run: reject(..., TOO_MANY_REJECTIONS)
      WITHOUT asking (don't nag the engineer)
    - NEW: already MAX_FAILED_WRITES failed writes: reject(..., TOO_MANY_FAILED_WRITES)
      WITHOUT asking
    - NEW: approved, but the tool returned an error (result["is_error"] is True,
      e.g. the content check found leaked markup): count it in run.failed_writes
    - otherwise ask: approved, reason = approve(block)
        yes -> result = run_tool(block, run); mark the recorded call approved
               (hint: run.calls[-1] is the call run_tool just recorded); return result
        no  -> count the rejection in run.rejections;
               return reject(block, run, rejection_message(reason))
    """
    if block.name not in NEEDS_APPROVAL:
        return run_tool(block, run)
    if run.rejections >= MAX_REJECTIONS:
        return reject(block, run, TOO_MANY_REJECTIONS)
    if run.failed_writes >= MAX_FAILED_WRITES:
        return reject(block, run, TOO_MANY_FAILED_WRITES)
    approved, reason = approve(block)
    if approved:
        result = run_tool(block, run)
        run.calls[-1].approved = True
        if result.get("is_error"):
            run.failed_writes += 1
        return result
    run.rejections += 1
    return reject(block, run, rejection_message(reason))


# ==============================================================================


def run_agent(
    alert: str, approve: Approver = ask_human, verbose: bool = True
) -> AgentRun:
    """Loop until Claude stops asking for tools. Returns the run, with its trace.

    verbose=False hides the step-by-step prints (the evals run 45 agents).
    """
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
                raise AgentError("Agent ended without an answer", run)
            return run
        if response.stop_reason != "tool_use":
            raise AgentError(f"Unexpected stop_reason: {response.stop_reason}", run)

        calls = [b for b in response.content if b.type == "tool_use"]
        if verbose:
            for b in calls:
                print(f"  step {step}: {b.name}({b.input})")
            for b in response.content:
                if b.type == "text" and b.text.strip():
                    print(f"  step {step} thinks: {b.text.strip()}")

        results = [handle_call(b, run, approve) for b in calls]
        messages.append({"role": "user", "content": results})

    raise AgentError(f"Agent did not finish within {MAX_STEPS} steps", run)


# ================================ TRY IT ======================================

if __name__ == "__main__":
    alert = " ".join(sys.argv[1:]) or "checkout-service: 5xx error rate at 12%"
    run = run_agent(alert)
    print(f"\n{run.answer}\n")
    print(
        f"{run.steps} API calls, {len(run.calls)} tool calls, "
        f"{run.input_tokens} input + {run.output_tokens} output tokens"
    )
