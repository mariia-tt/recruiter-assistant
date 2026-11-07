"""C2b LLM profiler: rich CandidateProfile for shortlisted CVs only (cached)."""

from recruiter import config, llm
from recruiter.schemas import CandidateProfile
from recruiter.skills import canonicalize

PROMPT = "cv_profiler"


def profile_cv(text: str) -> CandidateProfile:
    """Extract a CandidateProfile; skills are mapped to canonical dictionary names."""
    words = text.split()
    cv = " ".join(words[: config.LLM_MAX_CV_WORDS])
    messages = [
        {"role": "system", "content": llm.load_prompt(PROMPT)},
        {"role": "user", "content": cv},
    ]
    p = llm.structured(messages, CandidateProfile, component="cv_profiler")
    skills = list(dict.fromkeys(canonicalize(s) for s in p.skills if s.strip()))
    return p.model_copy(update={"skills": skills})
