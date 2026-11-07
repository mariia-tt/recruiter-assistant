from recruiter import profiler_llm
from recruiter.schemas import CandidateProfile


def test_profile_cv_canonicalizes_skills_and_truncates(monkeypatch):
    seen = {}

    def fake_structured(messages, schema, *, component, **kw):
        seen["words"] = len(messages[1]["content"].split())
        return CandidateProfile(headline="ACCOUNTANT", years_experience=5, seniority="senior",
                                skills=["MS Excel", "excel", "QuickBooks"])

    monkeypatch.setattr(profiler_llm.llm, "structured", fake_structured)
    p = profiler_llm.profile_cv("word " * 5000)
    assert p.skills == ["excel", "quickbooks"]
    assert seen["words"] == profiler_llm.config.LLM_MAX_CV_WORDS
