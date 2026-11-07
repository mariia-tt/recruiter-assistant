"""Shared IR / ranking metrics used by the eval scripts."""

import math

import numpy as np


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Share of relevant items that appear in the top-k."""
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant item (0 if none retrieved)."""
    for i, item in enumerate(ranked, 1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gains: dict[str, float], k: int) -> float:
    """NDCG with graded gains (2^g - 1), unlabelled items have gain 0."""

    def dcg(items: list[str]) -> float:
        return sum((2 ** gains.get(x, 0) - 1) / math.log2(i + 2) for i, x in enumerate(items[:k]))

    ideal = dcg(sorted(gains, key=gains.get, reverse=True))
    return dcg(ranked) / ideal if ideal > 0 else 0.0


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Share of the top-k that is relevant."""
    return len(set(ranked[:k]) & relevant) / k


def paired_bootstrap(a: list[float], b: list[float], n: int = 5000, seed: int = 42) -> dict:
    """Mean paired difference a - b with a 95 % bootstrap CI and win/tie/loss counts."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    return {
        "mean_diff": float(diff.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "wins_ties_losses": [int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum())],
        "n": len(diff),
    }
