"""C5 Outreach writer + C6 grounding checker.

The writer sees only facts from the LLM profile (C2b) and the scorecard; the checker
verifies every candidate claim against the full CV, and a claim counts as supported
only if its evidence quote is found verbatim in the CV (the LLM judge is not trusted
blindly). Failing messages are regenerated once with the problems as feedback.
"""

import re

from recruiter import config, llm
from recruiter.reranker_llm import quote_is_verbatim
from recruiter.schemas import (
    CandidateProfile,
    CandidateScore,
    Claim,
    GroundingReport,
    JobSpec,
    OutreachMessage,
    OutreachResult,
)

WRITER, CHECKER = "outreach_writer", "grounding_checker"
CTA_RE = re.compile(r"\?|\b(would you|are you open|let me know|interested in a)\b", re.IGNORECASE)


def candidate_facts(profile: CandidateProfile, score: CandidateScore | None) -> list[str]:
    """Bullet facts the writer may use (all come from the CV via C2b / C4)."""
    # Computed total years are deliberately NOT given: the CV never states them, so they
    # cannot be verified (C6 flagged them; C2b years MAE = 1.3 y in eval/results/profiler.json).
    facts = [f"Current/most recent title: {profile.headline}"]
    facts += [f"Role: {r.title}" + (f" ({r.period})" if r.period else "") for r in profile.recent_roles]
    facts += [f"Achievement: {a}" for a in profile.achievements]
    if score and score.matched_must:
        facts.append("Skills matching the vacancy: " + ", ".join(score.matched_must))
    elif profile.skills:
        facts.append("Skills: " + ", ".join(profile.skills[:8]))
    if profile.education:
        facts.append(f"Education: {profile.education}")
    if profile.domains:
        facts.append("Industries: " + ", ".join(profile.domains))
    return facts


def write_message(
    spec: JobSpec, facts: list[str], feedback: str | None = None
) -> OutreachMessage:
    """C5: one LLM call; optional feedback lists problems of the previous draft."""
    user = (f"## Vacancy\n{spec.model_dump_json(exclude_none=True)}\n\n## Candidate facts\n"
            + "\n".join(f"- {f}" for f in facts))
    if feedback:
        user += f"\n\n## Fix these problems of your previous draft\n{feedback}"
    messages = [{"role": "system", "content": llm.load_prompt(WRITER)},
                {"role": "user", "content": user}]
    return llm.structured(messages, OutreachMessage, component="outreach_writer", temperature=0.3)


def check_grounding(text: str, cv_text: str) -> GroundingReport:
    """C6: LLM lists candidate claims with evidence; evidence must be verbatim in the CV."""
    cv = " ".join(cv_text.split()[: config.LLM_MAX_CV_WORDS])
    messages = [{"role": "system", "content": llm.load_prompt(CHECKER)},
                {"role": "user", "content": f"## CV\n{cv}\n\n## Message\n{text}"}]
    report = llm.structured(messages, GroundingReport, component="grounding_checker")
    claims = [
        c if not c.supported or (c.evidence and quote_is_verbatim(c.evidence, cv_text))
        else Claim(claim=c.claim, supported=False, evidence=None)
        for c in report.claims
    ]
    return GroundingReport(claims=claims)


def constraint_violations(text: str) -> list[str]:
    """Deterministic checks of the business rules for a first-contact message."""
    problems = []
    n = len(text.split())
    if n > config.OUTREACH_MAX_WORDS:
        problems.append(f"too long: {n} words (max {config.OUTREACH_MAX_WORDS})")
    if "company name" in text.lower():
        problems.append('contains the anonymisation placeholder "Company Name"')
    if re.search(r"\[[^\]]*\]", text):
        problems.append("contains a [placeholder] in square brackets")
    if not CTA_RE.search(text):
        problems.append("no call to action / question at the end")
    return problems


def generate_outreach(
    spec: JobSpec, profile: CandidateProfile, cv_text: str, score: CandidateScore | None = None
) -> OutreachResult:
    """Write → check → (rewrite once with feedback) → return the best message and its report."""
    facts = candidate_facts(profile, score)
    feedback, history = None, []
    for attempt in range(1, config.OUTREACH_MAX_ATTEMPTS + 1):
        msg = write_message(spec, facts, feedback)
        report = check_grounding(msg.text, cv_text)
        violations = constraint_violations(msg.text)
        unsupported = [c.claim for c in report.claims if not c.supported]
        history.append({"text": msg.text, "n_claims": len(report.claims),
                        "unsupported": unsupported, "violations": violations})
        if not unsupported and not violations:
            return OutreachResult(message=msg, report=report, violations=[], attempts=attempt,
                                  passed=True, history=history)
        feedback = "\n".join([*(f"- unsupported claim, remove it: {u}" for u in unsupported),
                              *(f"- {v}" for v in violations)])
    return OutreachResult(message=msg, report=report, violations=violations, attempts=attempt,
                          passed=False, history=history)
