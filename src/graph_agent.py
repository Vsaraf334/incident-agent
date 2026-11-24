"""The incident agent rebuilt with LangGraph, to compare with the hand-written loop.

Same tools, same schemas (strict), same system prompt, same trace for the evals.
Only the control flow moves: the loop becomes a graph.

    START ─► agent ──(tool calls?)──yes─► tools ─┐
               ▲      │                           │
               │      no ─► END                   │
               └──────────────────────────────────┘

Run:  uv run --env-file .env python src/graph_agent.py "checkout-service: 5xx error rate at 12%"

Step 2 of the rebuild: no approval yet (file_incident runs without asking).
"""

import operator
import sys
from typing import Annotated, TypedDict, cast

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
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import agent  # reuse: TOOLS, FUNCTIONS, SYSTEM_PROMPT, MODEL, MAX_STEPS, ToolCall, AgentRun
from agent import FUNCTIONS, SYSTEM_PROMPT, TOOLS, AgentError, AgentRun, ToolCall


class State(TypedDict):
    """What every node can read. A node returns only the fields it changes.

    The Annotated[..., reducer] part says HOW an update merges into the state:
    - messages: add_messages appends new messages (like messages.append in the loop)
    - calls:    operator.add concatenates lists (old + new)
    A field with no reducer is simply overwritten.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    calls: Annotated[list[ToolCall], operator.add]


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


def tools_node(state: State) -> dict:
    """Run every tool call in the last AIMessage. Same rules as run_tool in agent.py.

    The last message is state["messages"][-1]; its .tool_calls is a list of dicts:
    {"name": "search_logs", "args": {"service": ..., "query": ...}, "id": "toolu_..."}

    For each tool call:
    - try: output = FUNCTIONS[call["name"]](**call["args"]); is_error = False
      except Exception as e: output = str(e); is_error = True
    - make ToolMessage(content=output, tool_call_id=call["id"],
                       status="error" if is_error else "success")
      (status="error" becomes is_error: true in the API request)
    - make ToolCall(call["name"], call["args"], output, is_error) for the trace

    Return {"messages": [the ToolMessages], "calls": [the ToolCalls]}
    """
    last_message = state["messages"][-1]  # Claude's reply that asked for the tools
    assert isinstance(last_message, AIMessage)  # only AIMessages have .tool_calls
    tool_messages = []  # results for Claude
    tool_calls = []  # records for the trace (the evals)

    for call in last_message.tool_calls:
        try:
            output = FUNCTIONS[call["name"]](**call["args"])
            is_error = False
        except Exception as e:  # noqa: BLE001 - any failure goes back to Claude as an error
            output = str(e)
            is_error = True

        tool_messages.append(
            ToolMessage(
                content=output,
                tool_call_id=call["id"],
                status="error" if is_error else "success",
            )
        )
        tool_calls.append(ToolCall(call["name"], call["args"], output, is_error))

    return {"messages": tool_messages, "calls": tool_calls}


def route(state: State) -> str:
    """The conditional edge after the agent node: where to go next?

    If the last message (an AIMessage) has tool_calls -> return "tools"
    otherwise -> return END   (END is the special node name "__end__")
    """
    last_message = state["messages"][-1]  # the reply agent_node just added
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# ==============================================================================


def build_graph():
    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile()


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


def run_graph(alert: str, verbose: bool = True) -> AgentRun:
    """Run the graph on one alert. Same contract as agent.run_agent."""
    graph = build_graph()
    start: State = {"messages": [HumanMessage(alert)], "calls": []}
    # Each node run is one "step" to LangGraph: MAX_STEPS API calls = 2 x MAX_STEPS nodes.
    config: RunnableConfig = {"recursion_limit": 2 * agent.MAX_STEPS}
    state = start
    try:
        # Two streams at once, as (mode, data) pairs after each node:
        #   "updates": {node_name: what that node returned}  -> to print progress
        #   "values":  the full state after the reducers ran  -> to keep
        for mode, data in graph.stream(
            start, config, stream_mode=["updates", "values"]
        ):
            if mode == "values":
                state = cast(State, data)
            elif verbose:
                for node, changes in cast(dict, data).items():
                    show(node, changes)
    except GraphRecursionError as e:
        raise AgentError(
            f"Agent did not finish within {agent.MAX_STEPS} steps",
            to_agent_run(alert, state),
        ) from e
    except RuntimeError as e:  # raised by agent_node (e.g. max_tokens)
        raise AgentError(str(e), to_agent_run(alert, state)) from e

    run = to_agent_run(alert, state)
    if not run.answer.strip():
        raise AgentError("Agent ended without an answer", run)
    return run


def show(node: str, changes: dict) -> None:
    """Print what a node did, like the loop's step lines."""
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
