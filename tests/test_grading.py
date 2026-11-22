"""Offline tests for the code graders, on hand-built traces."""

from agent import AgentRun, ToolCall
from grading import filed_incident, missing_checks, tools_used

INCIDENT = {"title": "T", "cause": "C", "evidence": "E", "severity": "SEV2"}


def trace(*calls: ToolCall) -> AgentRun:
    return AgentRun(alert="a", calls=list(calls))


def logs(service: str, is_error: bool = False) -> ToolCall:
    return ToolCall(
        "search_logs", {"service": service, "query": "ERROR"}, "...", is_error
    )


def deploys(service: str) -> ToolCall:
    return ToolCall("get_recent_deploys", {"service": service}, "...", False)


def incident(approved: bool | None, is_error: bool = False, **changes) -> ToolCall:
    return ToolCall(
        "file_incident", dict(INCIDENT, **changes), "...", is_error, approved
    )


# ------------------------------- filed_incident --------------------------------


def test_nothing_filed():
    assert filed_incident(trace(logs("a"), deploys("a"))) is None


def test_approved_and_successful_is_filed():
    assert filed_incident(trace(logs("a"), incident(approved=True))) == INCIDENT


def test_rejected_is_not_filed():
    assert filed_incident(trace(incident(approved=False, is_error=True))) is None


def test_approved_but_failed_is_not_filed():
    # e.g. the content checks rejected leaked markup after a human said yes
    assert filed_incident(trace(incident(approved=True, is_error=True))) is None


def test_last_filed_incident_wins():
    run = trace(incident(approved=True), incident(approved=True, severity="SEV1"))
    assert filed_incident(run)["severity"] == "SEV1"


# ------------------------------- missing_checks --------------------------------

MUST = [["search_logs", "auth-service"], ["get_recent_deploys", "auth-service"]]


def test_all_checks_done():
    assert (
        missing_checks(trace(logs("auth-service"), deploys("auth-service")), MUST) == []
    )


def test_missing_check_is_reported():
    assert missing_checks(trace(logs("auth-service")), MUST) == [
        ["get_recent_deploys", "auth-service"]
    ]


def test_wrong_service_does_not_count():
    run = trace(logs("checkout-service"), deploys("auth-service"))
    assert missing_checks(run, MUST) == [["search_logs", "auth-service"]]


def test_failed_call_does_not_count():
    run = trace(logs("auth-service", is_error=True), deploys("auth-service"))
    assert missing_checks(run, MUST) == [["search_logs", "auth-service"]]


def test_nothing_required():
    assert missing_checks(trace(), []) == []


# --------------------------------- tools_used ----------------------------------


def test_tools_used_counts_every_call():
    run = trace(logs("a"), logs("b", is_error=True), deploys("a"))
    assert tools_used(run) == {"search_logs": 2, "get_recent_deploys": 1}


def test_tools_used_empty():
    assert tools_used(trace()) == {}
