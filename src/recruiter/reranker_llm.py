"""C4b LLM re-ranker: interview-scorecard ratings for the top-N candidates in ONE call."""

import random
import re

from recruiter import config, llm
from recruiter.schemas import CandidateScore, JobSpec, LLMCandidateRating, LLMRerankResult

PROMPT = "llm_reranker"


def _excerpt(text: str) -> str:
    return " ".join(text.split()[: config.RERANK_CV_WORDS])


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def quote_is_verbatim(quote: str, cv_text: str) -> bool:
    """Evidence check: the quote must appear in the CV (case/punctuation-insensitive)."""
    q = _norm(quote)
    return bool(q) and q in _norm(cv_text)


def rate_candidates(
    spec: JobSpec, ids: list[str], texts: dict[str, str], seed: int = 0
) -> dict[str, LLMCandidateRating]:
    """Ask the LLM to rate candidates; order is shuffled (fixed seed) against position bias."""
    order = list(ids)
    random.Random(seed).shuffle(order)
    blocks = [f"### Candidate {i}\n{_excerpt(texts[cid])}" for i, cid in enumerate(order, 1)]
    user = f"## Vacancy\n{spec.model_dump_json(exclude_none=True)}\n\n## Candidates\n" + "\n\n".join(
        blocks
    )
    messages = [
        {"role": "system", "content": llm.load_prompt(PROMPT)},
        {"role": "user", "content": user},
    ]
    result = llm.structured(messages, LLMRerankResult, component="llm_reranker")
    return {order[r.candidate - 1]: r for r in result.ratings if 1 <= r.candidate <= len(order)}


def rerank(
    spec: JobSpec,
    ranked: list[CandidateScore],
    texts: dict[str, str],
    reorder: bool = config.RERANK_REORDER,
) -> tuple[list[CandidateScore], dict[str, LLMCandidateRating]]:
    """Rate the top-N; optionally reorder them by LLM `overall` (ties: scorecard total).

    Unverifiable evidence quotes are dropped so the UI never shows invented text.
    """
    head, tail = ranked[: config.RERANK_TOP_N], ranked[config.RERANK_TOP_N :]
    ratings = rate_candidates(spec, [c.id for c in head], texts)
    for cid, r in ratings.items():
        if not quote_is_verbatim(r.evidence_quote, texts[cid]):
            ratings[cid] = r.model_copy(update={"evidence_quote": ""})
    if reorder:
        head = sorted(head, key=lambda c: (-(ratings[c.id].overall if c.id in ratings else 0),
                                           -c.total))
    return head + tail, ratings
