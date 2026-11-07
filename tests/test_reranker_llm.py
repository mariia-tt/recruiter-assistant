from recruiter import reranker_llm
from recruiter.schemas import CandidateScore, JobSpec, LLMCandidateRating, LLMRerankResult


def _rating(n, overall):
    return LLMCandidateRating(candidate=n, skills_fit=3, experience_fit=3, role_fit=3,
                              overall=overall, evidence_quote="x")


def test_quote_is_verbatim_ignores_case_and_punctuation():
    cv = "Built ETL pipelines in PySpark, processing 2TB/day."
    assert reranker_llm.quote_is_verbatim("built ETL pipelines in pyspark processing", cv)
    assert not reranker_llm.quote_is_verbatim("built ML models", cv)
    assert not reranker_llm.quote_is_verbatim("", cv)


def test_rerank_maps_shuffled_numbers_back_and_sorts(monkeypatch):
    ids = ["a", "b", "c"]
    texts = {i: f"cv {i}" for i in ids}
    seen = {}

    def fake_structured(messages, schema, *, component, **kw):
        user = messages[1]["content"]
        # give the highest score to whichever number holds CV "c"
        numbers = {cid: n for n in (1, 2, 3) for cid in ids if f"Candidate {n}\ncv {cid}" in user}
        seen.update(numbers)
        return LLMRerankResult(ratings=[_rating(numbers[c], 5 if c == "c" else 2) for c in ids])

    monkeypatch.setattr(reranker_llm.llm, "structured", fake_structured)
    ranked = [CandidateScore(id=i, total=t, knocked_out=False, criteria=[])
              for i, t in zip(ids, (0.9, 0.8, 0.1), strict=True)]
    out, ratings = reranker_llm.rerank(JobSpec(title="x"), ranked, texts, reorder=True)
    assert [c.id for c in out] == ["c", "a", "b"]
    assert ratings["c"].overall == 5 and len(seen) == 3
    kept, _ = reranker_llm.rerank(JobSpec(title="x"), ranked, texts, reorder=False)
    assert [c.id for c in kept] == ["a", "b", "c"]
    assert ratings["a"].evidence_quote == ""  # "x" is not in "cv a" -> dropped
