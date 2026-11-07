"""JobSpec extractor: free text (JD or recruiter chat) -> structured vacancy. Used by C1."""

from recruiter import config, llm
from recruiter.schemas import JobSpec
from recruiter.skills import canonicalize

PROMPT = "jobspec_extractor"


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words])


def normalize(spec: JobSpec) -> JobSpec:
    """Map skills to canonical dictionary names and drop duplicates / overlaps."""
    must = list(dict.fromkeys(canonicalize(s) for s in spec.must_have_skills if s.strip()))
    nice = [s for s in dict.fromkeys(canonicalize(s) for s in spec.nice_to_have_skills if s.strip())
            if s not in must]
    return spec.model_copy(update={"must_have_skills": must, "nice_to_have_skills": nice})


def extract_jobspec(text: str) -> JobSpec:
    """Extract a normalised JobSpec from a job description or chat transcript."""
    messages = [
        {"role": "system", "content": llm.load_prompt(PROMPT)},
        {"role": "user", "content": _truncate(text, config.LLM_MAX_JD_WORDS)},
    ]
    return normalize(llm.structured(messages, JobSpec, component="jobspec_extractor"))
