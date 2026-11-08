"""Evaluate C4b LLM re-ranker on top-N of the scorecard v2 ranking (1 LLM call per JD).

Needs eval/results/ranker_features.csv (run eval/eval_ranker.py first).
Usage: uv run python eval/eval_reranker_llm.py --limit 2
"""

import argparse
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from recruiter import config, llm
from recruiter.metrics import ndcg_at_k, paired_bootstrap, precision_at_k
from recruiter.reranker_llm import PROMPT, quote_is_verbatim, rate_candidates
from recruiter.schemas import JobSpec

RATINGS = config.DATA_PROCESSED / "rerank_ratings.json"


def scorecard_order(g: pd.DataFrame) -> list[str]:
    w = config.SCORECARD_WEIGHTS
    score = sum(g[c] * v for c, v in w.items())
    return g.assign(_s=score).sort_values("_s", ascending=False).id.tolist()


def collect(df: pd.DataFrame, specs: dict, texts: dict, limit: int | None) -> dict:
    """Rate top-N of each JD (cached + saved after every JD so runs can resume)."""
    ratings = json.loads(RATINGS.read_text()) if RATINGS.exists() else {}
    jd_ids = sorted(df.jd_id.unique())[:limit]
    for jd_id in jd_ids:
        if jd_id in ratings:
            continue
        top = scorecard_order(df[df.jd_id == jd_id])[: config.RERANK_TOP_N]
        try:
            r = rate_candidates(JobSpec(**specs[jd_id]), top, texts)
        except llm.QuotaExhaustedError as e:
            print(f"STOP: {e}")
            break
        ratings[jd_id] = {cid: x.model_dump() for cid, x in r.items()}
        RATINGS.write_text(json.dumps(ratings, indent=1))
    return {k: v for k, v in ratings.items() if k in set(jd_ids)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(config.EVAL_DIR / "results" / "ranker_features.csv")
    fit = pd.read_parquet(config.DATA_PROCESSED / "fit_test.parquet")
    texts = dict(zip(fit.resume_id, fit.resume_text, strict=False))
    specs = json.loads(config.BENCHMARK_JOBSPECS.read_text())
    ratings = collect(df, specs, texts, args.limit)

    per = {"scorecard v2": [], "llm overall": [], "blend 50/50": []}
    p5 = {k: [] for k in per}
    overall, labels, verbatim, coverage = [], [], [], []
    for jd_id, r in ratings.items():
        g = df[df.jd_id == jd_id]
        gains = dict(zip(g.id, g.relevance, strict=True))
        good = set(g.id[g.relevance == 2])
        base = scorecard_order(g)
        head, tail = base[: config.RERANK_TOP_N], base[config.RERANK_TOP_N :]
        coverage.append(sum(c in r for c in head) / len(head))
        llm_head = sorted(head, key=lambda c: -(r[c]["overall"] if c in r else 0))
        # blend: scorecard rank position and LLM overall, both scaled to [0, 1]
        pos = {c: 1 - i / max(len(head) - 1, 1) for i, c in enumerate(head)}
        blend_head = sorted(
            head, key=lambda c: -(0.5 * pos[c] + 0.5 * ((r[c]["overall"] - 1) / 4 if c in r else 0)))
        for name, order in (("scorecard v2", base), ("llm overall", llm_head + tail),
                            ("blend 50/50", blend_head + tail)):
            per[name].append(ndcg_at_k(order, gains, 10))
            p5[name].append(precision_at_k(order, good, 5))
        for c, x in r.items():
            overall.append(x["overall"])
            labels.append(gains[c])
            verbatim.append(quote_is_verbatim(x["evidence_quote"], texts[c]))

    metrics = {k: {"ndcg@10": float(np.mean(v)), "p@5": float(np.mean(p5[k]))} for k, v in per.items()}
    out = {
        "component": "C4b LLM re-ranker",
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {"model": config.LLM_MODEL, "prompt_version": llm.prompt_version(PROMPT),
                   "top_n": config.RERANK_TOP_N, "cv_words": config.RERANK_CV_WORDS},
        "n": {"jds": len(ratings), "rated_candidates": len(overall)},
        "metrics": metrics,
        "significance": {
            f"{m} vs scorecard v2": paired_bootstrap(per[m], per["scorecard v2"])
            for m in ("llm overall", "blend 50/50")
        },
        "llm_overall_vs_label_spearman": float(spearmanr(overall, labels).statistic),
        "mean_overall_by_label": pd.Series(overall).groupby(labels).mean().round(2).to_dict(),
        "evidence_quote_verbatim_rate": float(np.mean(verbatim)),
        "rating_coverage": float(np.mean(coverage)),
        "llm_calls_used_today": llm.usage_today(config.LLM_MODEL),
    }
    (config.EVAL_DIR / "results" / "reranker_llm.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("n", "metrics", "llm_overall_vs_label_spearman",
                                          "mean_overall_by_label", "evidence_quote_verbatim_rate",
                                          "rating_coverage", "llm_calls_used_today")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
