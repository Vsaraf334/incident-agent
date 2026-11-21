import pytest

from tools import (
    file_incident,
    get_recent_deploys,
    known_services,
    search_logs,
)

# ------------------------------- the fake data --------------------------------


def test_every_service_with_logs_has_deploys():
    for service in known_services():
        assert "No deploys" not in get_recent_deploys(service)


# ----------------------------- get_recent_deploys -----------------------------


def test_deploys_newest_first():
    lines = get_recent_deploys("checkout-service").splitlines()
    assert [line.split()[0] for line in lines] == ["v2.4.1", "v2.4.0", "v2.3.9"]


def test_deploy_line_has_every_field():
    first = get_recent_deploys("checkout-service").splitlines()[0]
    for part in ("v2.4.1", "2026-11-20T14:02:00Z", "marco", "Switch payment client"):
        assert part in first


def test_deploys_respect_limit():
    lines = get_recent_deploys("payment-service", limit=2).splitlines()
    assert [line.split()[0] for line in lines] == ["v4.1.1", "v4.1.0"]


def test_deploys_only_for_that_service():
    assert "checkout" not in get_recent_deploys("payment-service").lower()


def test_deploys_unknown_service_lists_known_ones():
    with pytest.raises(ValueError, match="Known services: auth-service"):
        get_recent_deploys("checkout")


def test_no_deploys(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "new-service.log").write_text("")
    (tmp_path / "deploys.json").write_text("[]")
    assert get_recent_deploys("new-service", data_dir=tmp_path) == (
        "No deploys found for new-service."
    )


# -------------------------------- search_logs ---------------------------------


def test_search_is_case_insensitive():
    lines = search_logs("checkout-service", "timeoutexception").splitlines()
    assert len(lines) == 4
    assert all("TimeoutException" in line for line in lines)


def test_search_keeps_file_order():
    lines = search_logs("inventory-service", "ERROR").splitlines()
    assert lines[0].startswith("2026-11-20T09:15:02Z")
    times = [line.split()[0] for line in lines]
    assert times == sorted(times)


def test_search_respects_limit():
    assert len(search_logs("checkout-service", "ERROR", limit=3).splitlines()) == 3


def test_search_no_match():
    assert search_logs("notification-service", "ERROR") == (
        "No log lines in notification-service match 'ERROR'."
    )


def test_search_unknown_service():
    with pytest.raises(ValueError, match="Unknown service 'billing'"):
        search_logs("billing", "ERROR")


# ------------------------------- file_incident --------------------------------


def test_file_incident_writes_markdown(tmp_path):
    result = file_incident(
        "Checkout 502s",
        "v2.4.1 timeout too short",
        "14:03 TimeoutException",
        "SEV2",
        incidents_dir=tmp_path,
    )
    assert result == "Filed INC-001: Checkout 502s"
    text = (tmp_path / "INC-001.md").read_text()
    assert text.startswith("# INC-001: Checkout 502s")
    for part in (
        "- Severity: SEV2",
        "- Cause: v2.4.1 timeout too short",
        "## Evidence",
        "14:03 TimeoutException",
    ):
        assert part in text


def test_file_incident_numbers_increase(tmp_path):
    file_incident("A", "c", "e", "SEV3", incidents_dir=tmp_path)
    result = file_incident("B", "c", "e", "SEV3", incidents_dir=tmp_path)
    assert result == "Filed INC-002: B"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["INC-001.md", "INC-002.md"]


def test_file_incident_creates_the_folder(tmp_path):
    folder = tmp_path / "incidents"
    file_incident("A", "c", "e", "SEV1", incidents_dir=folder)
    assert (folder / "INC-001.md").exists()


def test_file_incident_bad_severity_writes_nothing(tmp_path):
    with pytest.raises(ValueError, match="SEV1, SEV2, SEV3"):
        file_incident("A", "c", "e", "high", incidents_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
