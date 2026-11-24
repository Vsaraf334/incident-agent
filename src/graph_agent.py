"""The incident agent rebuilt with LangGraph, to compare with the hand-written loop.

Same tools, same schemas (strict), same system prompt, same trace for the evals.
Only the control flow moves: the loop becomes a graph.

    START ─► agent ──(tool calls?)──yes─► approve ─► tools ─┐
               ▲      │                                       │
               │      no ─► END                               │
               └──────────────────────────────────────────────┘

approve pauses the graph (interrupt) for every write call; tools then runs the
approved calls and turns rejected ones into error results.

Run:  uv run --env-file .env python src/graph_agent.py "checkout-service: 5xx error rate at 12%"

"""

import operator
import sys
import uuid
from typing import Annotated, TypedDict, cast

from anthropic.types import ToolUseBlock
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

import agent  # reuse everything below; agent.MODEL is read at run time (the evals' --model)
from agent import (
    FUNCTIONS,
    MAX_FAILED_WRITES,
    MAX_REJECTIONS,
    NEEDS_APPROVAL,
    SYSTEM_PROMPT,
    TOO_MANY_FAILED_WRITES,
    TOO_MANY_REJECTIONS,
    TOOLS,
    AgentError,
    AgentRun,
    Approver,
    ToolCall,
    ask_human,
    rejection_message,
)


class State(TypedDict):
    """What every node can read. A node returns only the fields it changes.

    The Annotated[..., reducer] part says HOW an update merges into the state:
    - messages: add_messages appends new messages (like messages.append in the loop)
    - calls:    operator.add concatenates lists (old + new)
    A field with no reducer is simply overwritten.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    calls: Annotated[list[ToolCall], operator.add]
    # No reducer: each update REPLACES the old value.
    decisions: dict[str, dict]  # tool_call_id -> {"approved": bool, "message": str}
    rejections: int  # human "no"s so far
    failed_writes: int  # approved file_incident calls that the tool rejected


def make_llm() -> Runnable[LanguageModelInput, AIMessage]:
    """Claude with our tools bound to it. Built per run, so the evals' --model works."""
    llm = ChatAnthropic(
        model_name=agent.MODEL, max_tokens_to_sample=4096, timeout=None, stop=None
    )
    return llm.bind_tools(TOOLS)  # the same strict schemas as the loop


def agent_node(state: State) -> dict:
    """Call Claude with the system prompt + the conversation so far.

    1. response = make_llm().invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
    2. the loop's stop-reason guard: response.response_metadata.get("stop_reason")
       must be "end_turn" or "tool_use", otherwise raise RuntimeError(...)
       (e.g. max_tokens: a cut-off response may end in the middle of a tool call)
    3. return {"messages": [response]}     <- the reducer appends it
    """
    response = make_llm().invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
    stop_reason = response.response_metadata.get("stop_reason")
    if stop_reason not in ("end_turn", "tool_use"):
        raise RuntimeError(f"Unexpected stop_reason: {stop_reason}")

    return {"messages": [response]}


def approve_node(state: State) -> dict:
    """Ask a human about every write call in Claude's last reply. NO side effects here.

    interrupt() pauses the whole graph and hands the proposal to whoever is running it.
    When they resume with Command(resume=answer), this node starts again FROM THE TOP,
    and each interrupt() returns its stored answer instead of pausing. That is why this
    node only decides; the tools node, which runs after, is the one that writes.
    """
    last_message = state["messages"][-1]
    assert isinstance(last_message, AIMessage)
    decisions = {}
    rejections = state["rejections"]

    for call in last_message.tool_calls:
        if call["name"] not in NEEDS_APPROVAL:
            continue  # read tools need no approval
        if rejections >= MAX_REJECTIONS:
            decisions[call["id"]] = {"approved": False, "message": TOO_MANY_REJECTIONS}
            continue  # don't ask again
        if state["failed_writes"] >= MAX_FAILED_WRITES:
            decisions[call["id"]] = {
                "approved": False,
                "message": TOO_MANY_FAILED_WRITES,
            }
            continue

        answer = interrupt(
            call
        )  # pause here; resumes with {"approved": ..., "reason": ...}

        if answer["approved"]:
            decisions[call["id"]] = {"approved": True, "message": ""}
        else:
            rejections += 1
            decisions[call["id"]] = {
                "approved": False,
                "message": rejection_message(answer["reason"]),
            }

    return {"decisions": decisions, "rejections": rejections}


def tools_node(state: State) -> dict:
    """Run the tool calls in Claude's last reply, following the approval decisions.

    Read tools run as before. A write call runs only if approve_node approved it;
    a rejected one becomes an error result with the rejection message, and never runs.
    """
    last_message = state["messages"][-1]  # Claude's reply that asked for the tools
    assert isinstance(last_message, AIMessage)  # only AIMessages have .tool_calls
    tool_messages = []  # results for Claude
    tool_calls = []  # records for the trace (the evals)
    failed_writes = state["failed_writes"]

    for call in last_message.tool_calls:
        approved = None  # None: a read tool, no approval needed
        if call["name"] in NEEDS_APPROVAL:
            assert call["id"] is not None  # Claude's tool calls always have an id
            decision = state["decisions"][call["id"]]
            approved = decision["approved"]

        if approved is False:  # rejected: the tool never runs
            output, is_error = decision["message"], True
        else:
            try:
                output = FUNCTIONS[call["name"]](**call["args"])
                is_error = False
            except Exception as e:  # noqa: BLE001 - any failure goes back to Claude as an error
                output = str(e)
                is_error = True
            if approved and is_error:
                failed_writes += (
                    1  # approved, but the tool rejected it (e.g. leaked markup)
                )

        tool_messages.append(
            ToolMessage(
                content=output,
                tool_call_id=call["id"],
                status="error" if is_error else "success",
            )
        )
        tool_calls.append(
            ToolCall(call["name"], call["args"], output, is_error, approved)
        )

    return {
        "messages": tool_messages,
        "calls": tool_calls,
        "failed_writes": failed_writes,
    }


def route(state: State) -> str:
    """The conditional edge after the agent node: where to go next?

    If the last message (an AIMessage) has tool_calls -> return "approve"
    otherwise -> return END   (END is the special node name "__end__")
    """
    last_message = state["messages"][-1]  # the reply agent_node just added
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "approve"
    return END


# ==============================================================================


def build_graph():
    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("approve", approve_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["approve", END])
    graph.add_edge("approve", "tools")
    graph.add_edge("tools", "agent")
    # The checkpointer saves the state after every node. interrupt() needs it:
    # resuming means "load the saved state and continue from the paused node".
    # The checkpointer serializes the state. Our own ToolCall dataclass is in it, and
    # LangGraph only restores types it is told are safe (a tampered checkpoint must not
    # be able to create arbitrary objects), so we allow exactly that one class.
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("agent", "ToolCall")])
    return graph.compile(checkpointer=InMemorySaver(serde=serde))


def to_agent_run(alert: str, state: State) -> AgentRun:
    """Turn the final graph state into the same AgentRun the loop returns (for the evals)."""
    replies = [m for m in state["messages"] if isinstance(m, AIMessage)]
    run = AgentRun(alert=alert, calls=list(state["calls"]), steps=len(replies))
    for m in replies:
        if m.usage_metadata:
            run.input_tokens += m.usage_metadata["input_tokens"]
            run.output_tokens += m.usage_metadata["output_tokens"]
    run.answer = replies[-1].text if replies else ""
    return run


def run_graph(
    alert: str, approve: Approver = ask_human, verbose: bool = True
) -> AgentRun:
    """Run the graph on one alert. Same contract as agent.run_agent."""
    graph = build_graph()
    start: State = {
        "messages": [HumanMessage(alert)],
        "calls": [],
        "decisions": {},
        "rejections": 0,
        "failed_writes": 0,
    }
    config: RunnableConfig = {
        # Which saved conversation to continue. One id per run.
        "configurable": {"thread_id": str(uuid.uuid4())},
        # LangGraph counts node runs: 3 nodes per round (agent, approve, tools).
        "recursion_limit": 3 * agent.MAX_STEPS,
    }
    state = start
    graph_input: State | Command = start
    try:
        while True:
            pending = None  # the tool call the graph paused on, if any
            for mode, data in graph.stream(
                graph_input, config, stream_mode=["updates", "values"]
            ):
                if mode == "values":
                    state = cast(State, data)
                    continue
                for node, changes in cast(dict, data).items():
                    if node == "__interrupt__":  # the graph paused in approve_node
                        pending = changes[0].value
                    elif verbose:
                        show(node, changes)

            if pending is None:
                break  # the graph reached END
            # Ask the approver (a person, a test stub, or the evals' approve_all),
            # then resume the paused graph with the answer.
            block = ToolUseBlock(
                id=pending["id"],
                name=pending["name"],
                input=pending["args"],
                type="tool_use",
            )
            approved, reason = approve(block)
            graph_input = Command(resume={"approved": approved, "reason": reason})
    except GraphRecursionError as e:
        raise AgentError(
            f"Agent did not finish within {agent.MAX_STEPS} steps",
            to_agent_run(alert, state),
        ) from e
    except RuntimeError as e:  # raised by agent_node (e.g. max_tokens)
        raise AgentError(str(e), to_agent_run(alert, state)) from e

    run = to_agent_run(alert, state)
    run.rejections = state["rejections"]
    run.failed_writes = state["failed_writes"]
    if not run.answer.strip():
        raise AgentError("Agent ended without an answer", run)
    return run


def show(node: str, changes: dict) -> None:
    """Print what a node did, like the loop's step lines."""
    for decision in changes.get("decisions", {}).values():
        print(f"  [{node}] {'approved' if decision['approved'] else 'rejected'}")
    for m in changes.get("messages", []):
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                print(f"  [{node}] wants {call['name']}({call['args']})")
            if m.tool_calls and m.text.strip():
                print(f"  [{node}] thinks: {m.text.strip()}")
        elif isinstance(m, ToolMessage):
            first_line = str(m.content).splitlines()[0] if m.content else ""
            print(f"  [{node}] {m.status}: {first_line[:100]}")


if __name__ == "__main__":
    alert = " ".join(sys.argv[1:]) or "checkout-service: 5xx error rate at 12%"
    run = run_graph(alert)
    print(f"\n{run.answer}\n")
    print(
        f"{run.steps} API calls, {len(run.calls)} tool calls, "
        f"{run.input_tokens} input + {run.output_tokens} output tokens"
    )
