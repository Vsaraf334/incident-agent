"""Run every scenario in evals/scenarios.jsonl several times and score the agent.

For each run:
  task success = the filing decision is right AND the judge says the cause is "correct"
  tool choice  = every [tool, service] in must_check was called successfully
  steps        = API calls, tool calls and tokens (reported, not pass/fail)

A scenario passes if at least 2 of its 3 runs succeed. Gate: >= 10 of 15 scenarios.

Run:  uv run --env-file .env python evals/run_eval.py [--runs 3] [--only checkout-timeout]
"""

import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS.parent / "src"))

import agent
from grading import filed_incident, missing_checks, tools_used
from judge import judge
from tools import file_incident

SCENARIOS = EVALS / "scenarios.jsonl"
RESULTS = EVALS / "results.jsonl"  # git-ignored; the numbers go in the README
GATE = 10


def approve_all(_block) -> tuple[bool, str]:
    """The eval approver: say yes, so we measure Claude's OWN decision to file."""
    return True, ""


def run_one(scenario: dict) -> dict:
    """One agent run on one scenario, graded. Never raises: a crash is a failed run."""
    start = time.perf_counter()
    try:
        run = agent.run_agent(scenario["alert"], approve=approve_all, verbose=False)
        incident = filed_incident(run)
        verdict = judge(scenario, run.answer, incident["cause"] if incident else None)
    except Exception as e:  # noqa: BLE001 - one bad run must not stop the other 44
        return {
            "id": scenario["id"],
            "success": False,
            "crashed": f"{type(e).__name__}: {e}",
        }

    filed = incident is not None
    missing = missing_checks(run, scenario["must_check"])
    return {
        "id": scenario["id"],
        "success": filed == scenario["expect_filed"] and verdict.verdict == "correct",
        "decision_ok": filed == scenario["expect_filed"],
        "cause_ok": verdict.verdict == "correct",
        "tools_ok": not missing,
        "filed": filed,
        "verdict": verdict.verdict,
        "reason": verdict.reason,
        "missing": missing,
        "severity": incident["severity"] if incident else None,
        "steps": run.steps,
        "tool_calls": len(run.calls),
        "tools": tools_used(run),
        "tokens": run.input_tokens + run.output_tokens,
        "seconds": round(time.perf_counter() - start, 1),
        "answer": run.answer,
    }


def fraction(results: list[dict], key: str) -> str:
    """'2/3': how many results have a true value for key."""
    return f"{sum(bool(r.get(key)) for r in results)}/{len(results)}"


def average(results: list[dict], key: str) -> float:
    done = [r for r in results if "crashed" not in r]
    return sum(r[key] for r in done) / len(done) if done else 0.0


def report(scenarios: list[dict], results: list[dict], runs: int) -> None:
    need = runs // 2 + 1  # a majority: 2 of 3
    passed = 0
    print(
        f"\n{'scenario':<26} {'success':>7} {'decision':>8} {'cause':>5} {'tools':>5} "
        f"{'steps':>5} {'tokens':>7}"
    )
    for s in scenarios:
        rs = [r for r in results if r["id"] == s["id"]]
        wins = sum(r["success"] for r in rs)
        passed += wins >= need
        crashed = sum("crashed" in r for r in rs)
        note = f"  ({crashed} crashed)" if crashed else ""
        print(
            f"{s['id']:<26} {fraction(rs, 'success'):>7} {fraction(rs, 'decision_ok'):>8} "
            f"{fraction(rs, 'cause_ok'):>5} {fraction(rs, 'tools_ok'):>5} "
            f"{average(rs, 'steps'):>5.1f} {average(rs, 'tokens'):>7.0f}{note}"
        )

    done = [r for r in results if "crashed" not in r]
    print(
        f"\nScenarios passed (>= {need}/{runs} runs): {passed}/{len(scenarios)}"
        f"   gate >= {GATE}: {'PASS' if passed >= GATE else 'FAIL'}"
    )
    print(
        f"Runs: {fraction(results, 'success')} succeeded, tools ok {fraction(done, 'tools_ok')}, "
        f"{len(results) - len(done)} crashed"
    )
    if done:
        print(
            f"Cost per run: avg {average(done, 'steps'):.1f} API calls, "
            f"avg {average(done, 'tokens'):,.0f} tokens, "
            f"max {max(r['tokens'] for r in done):,} tokens"
        )
    severities = [f"{r['id']}={r['severity']}" for r in done if r["severity"]]
    print(f"Severities filed (reported, not graded): {', '.join(severities) or 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--only", help="run a single scenario id")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with SCENARIOS.open() as f:
        scenarios = [json.loads(line) for line in f if line.strip()]
    if args.only:
        scenarios = [s for s in scenarios if s["id"] == args.only]

    # Incidents filed during the eval go to a temporary folder, never to incidents/.
    tmp = tempfile.mkdtemp(prefix="eval-incidents-")
    agent.FUNCTIONS["file_incident"] = partial(file_incident, incidents_dir=Path(tmp))

    jobs = [s for s in scenarios for _ in range(args.runs)]
    print(
        f"{len(jobs)} runs ({len(scenarios)} scenarios x {args.runs}), "
        f"{args.workers} at a time..."
    )
    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:  # like an ExecutorService
        results = list(pool.map(run_one, jobs))

    with RESULTS.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    report(scenarios, results, args.runs)
    print(f"\nDetails (answers, judge reasons): {RESULTS.relative_to(EVALS.parent)}")


if __name__ == "__main__":
    main()
