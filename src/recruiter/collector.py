"""C1 Requirements collector: LLM slot extraction from the whole chat + rule-based follow-ups.

Re-extracting from the full transcript every turn keeps the collector stateless and lets the
recruiter correct earlier answers. Which question to ask next is decided by rules, not the
LLM, so behaviour is predictable and costs one LLM call per recruiter message.
"""

from recruiter import config
from recruiter.jobspec import extract_jobspec
from recruiter.schemas import CollectorTurn, JobSpec

QUESTIONS = {
    "title": "What role are you hiring for (job title)?",
    "must_have_skills": "Which skills or tools are must-haves for this role?",
    "min_years": "How many years of relevant experience at minimum?",
    "seniority": "What level is it: junior, mid, senior or lead?",
}
OPTIONAL_HINT = (
    "Optionally: nice-to-have skills, industry/domain, or location — "
    "or say 'search' to start now."
)


def transcript(history: list[dict]) -> str:
    """Recruiter messages only; assistant questions add no facts."""
    return "\n".join(f"Recruiter: {m['content']}" for m in history if m["role"] == "user")


def summarize(spec: JobSpec) -> str:
    """Human-readable confirmation of what was understood."""
    parts = [f"**{spec.title}**"]
    if spec.seniority:
        parts.append(spec.seniority)
    if spec.min_years is not None:
        parts.append(f"{spec.min_years:g}+ years")
    text = ", ".join(parts)
    if spec.must_have_skills:
        text += f"; must-have: {', '.join(spec.must_have_skills)}"
    if spec.nice_to_have_skills:
        text += f"; nice-to-have: {', '.join(spec.nice_to_have_skills)}"
    if spec.domain:
        text += f"; domain: {spec.domain}"
    return text


def next_turn(history: list[dict]) -> CollectorTurn:
    """Update the JobSpec from the chat and decide: ask follow-ups or start the search."""
    spec = extract_jobspec(transcript(history))
    missing = spec.missing_mandatory()
    n_user = sum(m["role"] == "user" for m in history)
    wants_search = "search" in history[-1]["content"].lower()
    if not missing or n_user >= config.COLLECTOR_MAX_TURNS or (wants_search and spec.title):
        reply = f"Got it: {summarize(spec)}. Searching the candidate database…"
        return CollectorTurn(spec=spec, missing=missing, reply=reply, done=True)
    questions = " ".join(QUESTIONS[s] for s in missing[:2])
    known = f"So far: {summarize(spec)}. " if spec.title else ""
    hint = f"\n\n_{OPTIONAL_HINT}_" if len(missing) <= 2 else ""
    return CollectorTurn(spec=spec, missing=missing, reply=known + questions + hint, done=False)
