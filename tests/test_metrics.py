import pytest

from recruiter.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def test_recall_and_rr():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"b", "z"}, 2) == 0.5
    assert reciprocal_rank(ranked, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"z"}) == 0.0
    assert precision_at_k(ranked, {"a", "b"}, 4) == 0.5


def test_ndcg_perfect_and_reversed():
    gains = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], gains, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["c", "b", "a"], gains, 3) < 0.7
