"""Grade the agent's conclusion with Claude (the week 6 pattern).

Code checks the facts in the trace (filed or not, which tools ran). The judge
only answers the question code can't: does the answer state the right cause?
"""

from typing import Literal

import anthropic
from pydantic import BaseModel

JUDGE_MODEL = "claude-sonnet-5-5"  # a stronger model grades the cheaper agent
JUDGE_PROMPT = """You grade the conclusion of an on-call agent that investigated an alert.

<alert>{alert}</alert>
<incident_expected>{expect_filed}</incident_expected>
<expected_cause>{expected_cause}</expected_cause>
<must_not_blame>{must_not_blame}</must_not_blame>
<agent_answer>{answer}</agent_answer>
<filed_incident_cause>{filed_cause}</filed_incident_cause>

Judge only the cause the agent concludes, not its wording, severity or formatting.
- "correct": the agent's conclusion matches the expected cause in substance. If no
  incident was expected, the agent concludes there is no clear incident.
- "partial": the agent mentions the right cause but doesn't commit to it, or commits
  to it next to a wrong cause.
- "incorrect": the agent concludes a different cause, blames what must_not_blame
  names, or claims an incident where none was expected.
Mentioning must_not_blame only to rule it out is fine.

Reply with only JSON: {{"verdict": "...", "reason": "one sentence"}}"""

client = anthropic.Anthropic()


class Verdict(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"]
    reason: str


def judge(scenario: dict, answer: str, filed_cause: str | None) -> Verdict:
    prompt = JUDGE_PROMPT.format(
        alert=scenario["alert"],
        expect_filed=scenario["expect_filed"],
        expected_cause=scenario["expected_cause"],
        must_not_blame=scenario["must_not_blame"] or "(nothing specific)",
        answer=answer,
        filed_cause=filed_cause or "(no incident filed)",
    )
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Judge reply was cut off; increase max_tokens")
    text = "".join(b.text for b in response.content if b.type == "text")
    start, end = text.find("{"), text.rfind("}")  # skip any sentence before the JSON
    if start == -1 or end == -1:
        raise ValueError(f"Judge reply has no JSON: {text!r}")
    return Verdict.model_validate_json(text[start : end + 1])
