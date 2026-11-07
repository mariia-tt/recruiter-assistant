from recruiter import collector
from recruiter.schemas import JobSpec


def _fake(spec):
    return lambda text: spec


def test_asks_for_missing_slots(monkeypatch):
    monkeypatch.setattr(collector, "extract_jobspec", _fake(JobSpec(title="Data Engineer")))
    turn = collector.next_turn([{"role": "user", "content": "Need a data engineer"}])
    assert not turn.done
    assert turn.missing == ["must_have_skills", "min_years", "seniority"]
    assert "must-haves" in turn.reply and "years" in turn.reply  # max 2 questions


def test_done_when_complete(monkeypatch):
    spec = JobSpec(title="Data Engineer", must_have_skills=["spark"], min_years=3, seniority="mid")
    monkeypatch.setattr(collector, "extract_jobspec", _fake(spec))
    turn = collector.next_turn([{"role": "user", "content": "..."}])
    assert turn.done and "Searching" in turn.reply


def test_search_keyword_and_turn_limit(monkeypatch):
    monkeypatch.setattr(collector, "extract_jobspec", _fake(JobSpec(title="Chef")))
    assert collector.next_turn([{"role": "user", "content": "chef, just search"}]).done
    history = [{"role": "user", "content": "x"}] * collector.config.COLLECTOR_MAX_TURNS
    assert collector.next_turn(history).done


def test_transcript_uses_only_recruiter_messages():
    h = [{"role": "assistant", "content": "Q?"}, {"role": "user", "content": "A"}]
    assert collector.transcript(h) == "Recruiter: A"
