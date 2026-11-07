"""C4 Scorecard ranker: transparent weighted criteria + knock-out rule + evidence.

Business logic is documented in docs/01_business_understanding.md.
"""

import numpy as np

from recruiter import config
from recruiter.retrieval import HybridIndex, dense_scores, embed, query_from_spec
from recruiter.schemas import CandidateScore, CriterionScore, JobSpec, RuleProfile
from recruiter.skills import find_skill

LEVELS = ["junior", "mid", "senior", "lead"]


def _snippet(text: str, skill: str, width: int = 60) -> str | None:
    """Short CV fragment around the first mention of a skill (evidence for the recruiter)."""
    m = find_skill(skill, text)
    if not m:
        return None
    s, e = max(m.start() - width, 0), min(m.end() + width, len(text))
    return "…" + " ".join(text[s:e].split()) + "…"


def _coverage(skills: list[str], text: str) -> tuple[float, list[str], list[str], list[str]]:
    """Share of skills found, plus matched, missing and evidence lists."""
    if not skills:
        return 1.0, [], [], []
    matched, missing, evidence = [], [], []
    for s in skills:
        snip = _snippet(text, s)
        if snip:
            matched.append(s)
            evidence.append(f"{s}: {snip}")
        else:
            missing.append(s)
    return len(matched) / len(skills), matched, missing, evidence


def rescale_sim(sim: float) -> float:
    """Map a cosine similarity to [0, 1] using the empirical range of the encoder."""
    return float(np.clip((sim - config.SIM_LOW) / (config.SIM_HIGH - config.SIM_LOW), 0, 1))


def experience_score(years: float, min_years: float | None) -> float:
    """1 if the requirement is met; linear penalty below it; 1 when not required."""
    if not min_years:
        return 1.0
    return float(min(years / min_years, 1.0))


def seniority_score(candidate: str, required: str | None) -> float:
    """Exact level 1.0, adjacent level 0.5, otherwise 0; 1 when not required."""
    if required is None:
        return 1.0
    gap = abs(LEVELS.index(candidate) - LEVELS.index(required))
    return {0: 1.0, 1: 0.5}.get(gap, 0.0)


def score_candidate(
    spec: JobSpec,
    profile: RuleProfile,
    text: str,
    title_sim: float,
    domain_sim: float,
    semantic_sim: float = 0.0,
    weights: dict[str, float] | None = None,
    knockout: float | None = None,
) -> CandidateScore:
    """Compute every scorecard criterion for one candidate and the weighted total."""
    weights = weights or config.SCORECARD_WEIGHTS
    must, matched, missing, must_ev = _coverage(spec.must_have_skills, text)
    nice, _, _, nice_ev = _coverage(spec.nice_to_have_skills, text)
    values = {
        "must_have": (must, must_ev),
        "experience": (experience_score(profile.years_experience, spec.min_years),
                       [f"{profile.years_experience} years in CV vs {spec.min_years} required"]),
        "title": (rescale_sim(title_sim) if spec.title else 1.0,
                  [f"CV headline: {profile.headline}"]),
        "nice_to_have": (nice, nice_ev),
        "seniority": (seniority_score(profile.seniority, spec.seniority),
                      [f"{profile.seniority} vs {spec.seniority} required"]),
        "domain": (rescale_sim(domain_sim) if spec.domain else 1.0, []),
        "semantic": (rescale_sim(semantic_sim), []),
    }
    criteria = [
        CriterionScore(name=n, score=round(values[n][0], 3), weight=w, evidence=values[n][1])
        for n, w in weights.items()
    ]
    total = sum(c.score * c.weight for c in criteria) / (sum(weights.values()) or 1)
    return CandidateScore(
        id=profile.id,
        total=round(total, 4),
        knocked_out=knockout is not None and must < knockout,
        criteria=criteria,
        matched_must=matched,
        missing_must=missing,
    )


def similarity_features(
    spec: JobSpec, profiles: list[RuleProfile], index: HybridIndex
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Cosine similarities: CV headline vs job title, CV chunks vs domain, CV vs whole spec."""
    title_sim = {p.id: 0.0 for p in profiles}
    domain_sim = {p.id: 0.0 for p in profiles}
    pos = {doc_id: i for i, doc_id in enumerate(index.ids)}
    dense = dense_scores(index, query_from_spec(spec))
    semantic_sim = {p.id: float(dense[pos[p.id]]) for p in profiles}
    if spec.title:
        q = embed([spec.title])[0]
        heads = embed([p.headline or "unknown" for p in profiles])
        title_sim = {p.id: float(s) for p, s in zip(profiles, heads @ q, strict=True)}
    if spec.domain:
        sims = index.chunk_emb @ embed([spec.domain], is_query=True)[0]
        for p in profiles:
            domain_sim[p.id] = float(sims[index.chunk_owner == pos[p.id]].max())
    return title_sim, domain_sim, semantic_sim


def rank(
    spec: JobSpec,
    profiles: list[RuleProfile],
    texts: dict[str, str],
    index: HybridIndex,
    weights: dict[str, float] | None = None,
    knockout: float | None = config.KNOCKOUT_MUST_COVERAGE,
) -> list[CandidateScore]:
    """Score candidates; shortlist (passing the knock-out rule, if on) first, then near misses."""
    title_sim, domain_sim, semantic_sim = similarity_features(spec, profiles, index)
    scored = [
        score_candidate(spec, p, texts[p.id], title_sim[p.id], domain_sim[p.id],
                        semantic_sim[p.id], weights, knockout)
        for p in profiles
    ]
    return sorted(scored, key=lambda c: (c.knocked_out, -c.total))
