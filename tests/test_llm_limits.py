import time

import pytest

from recruiter import config, llm


@pytest.fixture(autouse=True)
def isolated_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "USAGE_FILE", tmp_path / "usage.json")
    monkeypatch.setattr(config, "LLM_MODEL", "main")
    monkeypatch.setattr(config, "LLM_RPM", 2)
    monkeypatch.setattr(config, "LLM_RPD", 3)


def test_daily_budget_stops_requests(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    for _ in range(3):
        llm._acquire("main")
    with pytest.raises(llm.QuotaExhaustedError):
        llm._acquire("main")
    assert llm.usage_today("main") == 3


def test_sliding_window_waits(monkeypatch):
    slept = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    llm._acquire("main")
    llm._acquire("main")
    llm._acquire("main")  # third request inside 60 s with RPM=2 -> must wait
    assert len(slept) == 1 and 55 < slept[0] <= 61


def test_usage_persists_between_runs():
    llm._acquire("main")
    assert config.USAGE_FILE.exists()
    stamps = llm._load_usage()["main"]
    assert len(stamps) == 1 and abs(stamps[0] - time.time()) < 5
