from recruiter import ranker
from recruiter.schemas import JobSpec, RuleProfile

SPEC = JobSpec(title="Data Engineer", seniority="senior", must_have_skills=["python", "spark"],
               nice_to_have_skills=["aws"], min_years=5)


def _profile(cid, years=6.0, level="senior"):
    return RuleProfile(id=cid, headline="DATA ENGINEER", skills=[], years_experience=years,
                       seniority=level)


def test_component_scores():
    assert ranker.experience_score(2.5, 5) == 0.5
    assert ranker.experience_score(10, 5) == 1.0
    assert ranker.experience_score(1, None) == 1.0
    assert ranker.seniority_score("mid", "senior") == 0.5
    assert ranker.seniority_score("junior", "lead") == 0.0
    assert ranker.rescale_sim(0.9) == 1.0 and ranker.rescale_sim(0.3) == 0.0


def test_full_match_beats_partial_and_has_evidence():
    full = ranker.score_candidate(SPEC, _profile("a"), "Built pipelines in PySpark and pandas on AWS",
                                  0.9, 0.9)
    partial = ranker.score_candidate(SPEC, _profile("b"), "Python scripts only", 0.9, 0.9)
    assert full.total > partial.total
    assert full.matched_must == ["python", "spark"] and not full.knocked_out
    assert any("pyspark" in e.lower() for c in full.criteria for e in c.evidence)
    assert partial.missing_must == ["spark"] and not partial.knocked_out


def test_knockout_only_when_enabled():
    off = ranker.score_candidate(SPEC, _profile("c"), "Excel reports", 0.9, 0.9)
    on = ranker.score_candidate(SPEC, _profile("c"), "Excel reports", 0.9, 0.9, knockout=0.5)
    assert not off.knocked_out and on.knocked_out
    assert on.missing_must == ["python", "spark"]


def test_weights_sum_and_total_in_unit_range():
    c = ranker.score_candidate(SPEC, _profile("a"), "python spark aws", 0.9, 0.9)
    assert abs(sum(x.weight for x in c.criteria) - 1) < 1e-9
    assert 0 <= c.total <= 1
