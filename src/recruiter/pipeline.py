"""End-to-end pipeline over the candidate DB: JobSpec -> C3 retrieve -> C4 rank -> C4b explain."""

import json
import logging
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from recruiter import config
from recruiter.outreach import generate_outreach
from recruiter.profiler_llm import profile_cv
from recruiter.profiler_rules import profile
from recruiter.ranker import rank
from recruiter.reranker_llm import rerank
from recruiter.retrieval import HybridIndex, build_index, query_from_spec, search
from recruiter.schemas import (
    CandidateScore,
    JobSpec,
    LLMCandidateRating,
    OutreachResult,
    RuleProfile,
)

log = logging.getLogger(__name__)
PROFILES_FILE = config.DATA_PROCESSED / "rule_profiles.json"


@dataclass
class CandidateDB:
    texts: dict[str, str]
    categories: dict[str, str]
    profiles: dict[str, RuleProfile]
    index: HybridIndex


@lru_cache(maxsize=1)
def load_db() -> CandidateDB:
    """Load cleaned CVs, rule profiles (C2a, cached) and the hybrid index (C3, cached)."""
    df = pd.read_parquet(config.DATA_PROCESSED / "resumes.parquet")
    texts = dict(zip(df.id, df.text, strict=True))
    if PROFILES_FILE.exists():
        profiles = {k: RuleProfile(**v) for k, v in json.loads(PROFILES_FILE.read_text()).items()}
    else:
        profiles = {cid: profile(cid, t) for cid, t in texts.items()}
        PROFILES_FILE.write_text(json.dumps({k: v.model_dump() for k, v in profiles.items()}))
    index = build_index(df.id.tolist(), df.text.tolist())
    return CandidateDB(texts, dict(zip(df.id, df.category, strict=True)), profiles, index)


def search_candidates(
    spec: JobSpec, knockout: float | None = config.KNOCKOUT_MUST_COVERAGE
) -> list[CandidateScore]:
    """Retrieve top-K with the hybrid retriever and rank them with the scorecard."""
    db = load_db()
    hits = search(db.index, query_from_spec(spec), k=config.RETRIEVE_K)
    pool = [db.profiles[h.id] for h in hits]
    ranked = rank(spec, pool, db.texts, db.index, knockout=knockout)
    log.info("search: %d retrieved, top=%s", len(hits), ranked[0].id if ranked else None)
    return ranked


def explain_top(
    spec: JobSpec, ranked: list[CandidateScore]
) -> tuple[list[CandidateScore], dict[str, LLMCandidateRating]]:
    """One LLM call: recruiter-style ratings + verified evidence quotes for the top-N."""
    return rerank(spec, ranked, load_db().texts)


def write_outreach(spec: JobSpec, score: CandidateScore) -> OutreachResult:
    """C2b profile + C5 writer + C6 checker for one shortlisted candidate (2-5 LLM calls)."""
    text = load_db().texts[score.id]
    return generate_outreach(spec, profile_cv(text), text, score)
