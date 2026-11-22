"""Code graders: facts read straight from the agent's trace (run.calls). No LLM."""

from collections import Counter

from agent import AgentRun


def filed_incident(run: AgentRun) -> dict | None:
    """The input of the incident that was actually filed, or None.

    "Actually filed" means: a call named "file_incident" that was approved
    (call.approved is True) AND did not fail (call.is_error is False). A call
    rejected by the content checks was approved but failed, so it doesn't count.
    If several were filed, return the last one.
    """
    for call in reversed(run.calls):
        if call.name == "file_incident" and call.approved and not call.is_error:
            return call.input
    return None


def missing_checks(run: AgentRun, must_check: list[list[str]]) -> list[list[str]]:
    """The [tool, service] pairs from must_check that the agent never called successfully.

    must_check looks like [["search_logs", "auth-service"], ["get_recent_deploys", "auth-service"]].
    A call counts if call.name == tool, call.input.get("service") == service,
    and call.is_error is False (a call that failed showed the agent nothing).
    Return the pairs that no call matched, in the order given.
    """
    return [
        [tool, service]
        for tool, service in must_check
        if not any(
            call.name == tool
            and call.input.get("service") == service
            and not call.is_error
            for call in run.calls
        )
    ]


def tools_used(run: AgentRun) -> dict[str, int]:
    """How many times each tool was called, e.g. {"search_logs": 3, "get_recent_deploys": 1}.

    Count every call, including failed ones: this is about cost, not correctness.
    """
    return dict(Counter(call.name for call in run.calls))
