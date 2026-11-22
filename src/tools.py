"""The agent's three tools, as plain Python functions over the fake data in data/.

Each tool returns text, because the text is exactly what Claude will read.
Bad input raises ValueError with a message that tells Claude how to fix the call.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INCIDENTS_DIR = ROOT / "incidents"
SEVERITIES = ("SEV1", "SEV2", "SEV3")
MAX_DEPLOYS = 10
MAX_LOG_LINES = 50

# The model sometimes leaks its own tool-call format into a long text field,
# e.g. 'cause': '...</cause>\n<parameter name="evidence">...'. Such a field is
# rejected, so a garbled incident is never written.
MARKUP = (
    "<parameter",
    "</parameter>",
    "</invoke>",
    "</title>",
    "</cause>",
    "</evidence>",
)


def clamp(value: int, low: int, high: int) -> int:
    """Force value into [low, high]. A limit of 0 or -1 would silently drop
    results (lines[:-1] loses the last line), so out-of-range limits are fixed,
    not trusted."""
    return max(low, min(value, high))


def load_deploys(data_dir: Path = DATA_DIR) -> list[dict]:
    """All deploys from data/deploys.json, in file order (oldest first)."""
    return json.loads((data_dir / "deploys.json").read_text())


def known_services(data_dir: Path = DATA_DIR) -> list[str]:
    """Every service that has a log file, sorted."""
    return sorted(p.stem for p in (data_dir / "logs").glob("*.log"))


def check_service(service: str, data_dir: Path = DATA_DIR) -> None:
    """Raise ValueError if the service doesn't exist, listing the ones that do."""
    services = known_services(data_dir)
    if service not in services:
        raise ValueError(
            f"Unknown service '{service}'. Known services: {', '.join(services)}"
        )


def get_recent_deploys(service: str, limit: int = 3, data_dir: Path = DATA_DIR) -> str:
    """The service's last `limit` deploys, newest first, one per line:

    v2.4.1  2026-11-20T14:02:00Z  marco  Switch payment client to async calls ...

    If the service has no deploys: "No deploys found for <service>."
    """
    check_service(service, data_dir)
    deploys = [d for d in load_deploys(data_dir) if d["service"] == service]
    limit = clamp(limit, 1, MAX_DEPLOYS)
    deploys = sorted(deploys, key=lambda d: d["time"], reverse=True)[:limit]
    if not deploys:
        return f"No deploys found for {service}."
    return "\n".join(
        f"{d['version']}  {d['time']}  {d['author']}  {d['change']}" for d in deploys
    )


def search_logs(
    service: str, query: str, limit: int = 20, data_dir: Path = DATA_DIR
) -> str:
    """Log lines of the service that contain `query` (case-insensitive), in file order.

    At most `limit` lines, joined with "\\n".
    If nothing matches: "No log lines in <service> match '<query>'."
    """
    check_service(service, data_dir)
    limit = clamp(limit, 1, MAX_LOG_LINES)
    lines = (data_dir / "logs" / f"{service}.log").read_text().splitlines()
    matching_lines = [line for line in lines if query.lower() in line.lower()][:limit]
    if not matching_lines:
        return f"No log lines in {service} match '{query}'."
    return "\n".join(matching_lines)


def file_incident(
    title: str,
    cause: str,
    evidence: str,
    severity: str,
    incidents_dir: Path = INCIDENTS_DIR,
) -> str:
    """Write the incident to incidents/INC-001.md (002, 003, ...) and return
    "Filed INC-001: <title>".

    The file looks like:

    # INC-001: <title>

    - Severity: SEV2
    - Cause: <cause>

    ## Evidence

    <evidence>
    """
    # Validate before writing anything: a rejected call must not leave a file behind.
    if severity not in SEVERITIES:
        raise ValueError(
            f"Invalid severity '{severity}'. Allowed values: {', '.join(SEVERITIES)}"
        )
    for name, value in {"title": title, "cause": cause, "evidence": evidence}.items():
        if not value.strip():
            raise ValueError(f"The {name} field is empty. Nothing was filed.")
        if any(marker in value for marker in MARKUP):
            raise ValueError(
                f"The {name} field contains tool-call markup such as "
                "'<parameter name=...>'. Nothing was filed. Call file_incident again "
                "with each value in its own field and plain text only."
            )
    incidents_dir.mkdir(parents=True, exist_ok=True)
    number = len(list(incidents_dir.glob("INC-*.md"))) + 1
    incident_id = f"INC-{number:03d}"
    (incidents_dir / f"{incident_id}.md").write_text(f"""# {incident_id}: {title}

- Severity: {severity}
- Cause: {cause}

## Evidence

{evidence}
""")
    return f"Filed {incident_id}: {title}"
