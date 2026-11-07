from recruiter import jobspec
from recruiter.schemas import JobSpec


def test_normalize_canonicalizes_and_dedupes():
    spec = JobSpec(must_have_skills=["Python", "pandas", "SQL"], nice_to_have_skills=["sql", "Tableau"])
    out = jobspec.normalize(spec)
    assert out.must_have_skills == ["python", "sql"]
    assert out.nice_to_have_skills == ["tableau"]
    spec = JobSpec(must_have_skills=["Apache Spark", "net", "object-oriented programming"])
    assert jobspec.normalize(spec).must_have_skills == ["spark", ".net", "oop"]


def test_extract_jobspec_uses_llm_structured(monkeypatch):
    captured = {}

    def fake_structured(messages, schema, *, component, **kw):
        captured["system"] = messages[0]["content"]
        return schema(title="Data Engineer", must_have_skills=["PySpark"], min_years=5)

    monkeypatch.setattr(jobspec.llm, "structured", fake_structured)
    spec = jobspec.extract_jobspec("We need a data engineer with PySpark, 5+ years.")
    assert spec.must_have_skills == ["spark"]
    assert "JobSpec" in captured["system"]
    assert spec.missing_mandatory() == ["seniority"]
