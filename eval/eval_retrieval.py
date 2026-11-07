"""Evaluate C3 retriever on the resume–JD fit benchmark (no LLM calls).

Two views:
- corpus retrieval: search all benchmark resumes with the JD; relevant = "Good Fit"
  resumes of that JD -> Recall@k, MRR (JDs with >= 1 Good Fit);
- pool ranking: rerank only the labelled resumes of each JD -> NDCG@10 with graded
  labels (0/1/2), over JDs with >= 1 non-zero label. Baseline for the C4 ranker.
Query = raw JD text (JobSpec-based queries are evaluated once C1 exists).
"""

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from recruiter import config
from recruiter.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from recruiter.retrieval import build_index, search

KS = (10, 20, 50)
MODES = ("bm25", "dense", "hybrid")
SEED = 42


def random_baseline(fit: pd.DataFrame, n_docs: int, rng: np.random.Generator) -> dict:
    """Expected scores of a random ordering, estimated with 200 shuffles per JD."""
    rec = {k: [] for k in KS}
    ndcg = []
    for _, g in fit.groupby("jd_id"):
        rel = set(g.resume_id[g.relevance == 2])
        gains = dict(zip(g.resume_id, g.relevance, strict=True))
        for _ in range(200):
            if rel:
                for k in KS:
                    rec[k].append(min(k, n_docs) * len(rel) / n_docs / len(rel))
            if g.relevance.max() > 0:
                ndcg.append(ndcg_at_k(list(rng.permutation(g.resume_id)), gains, 10))
    return {**{f"recall@{k}": float(np.mean(v)) for k, v in rec.items()},
            "pool_ndcg@10": float(np.mean(ndcg))}


def main() -> None:
    fit = pd.read_parquet(config.DATA_PROCESSED / "fit_test.parquet")
    docs = fit.drop_duplicates("resume_id")
    index = build_index(docs.resume_id.tolist(), docs.resume_text.tolist())
    jds = fit.drop_duplicates("jd_id").set_index("jd_id").jd_text

    results, per_jd = {}, []
    for mode in MODES:
        rec = {k: [] for k in KS}
        rr, ndcg = [], []
        for jd_id, g in fit.groupby("jd_id"):
            rel = set(g.resume_id[g.relevance == 2])
            gains = dict(zip(g.resume_id, g.relevance, strict=True))
            if rel:
                ranked = [h.id for h in search(index, jds[jd_id], k=max(KS), mode=mode)]
                for k in KS:
                    rec[k].append(recall_at_k(ranked, rel, k))
                rr.append(reciprocal_rank(ranked, rel))
            if g.relevance.max() > 0:
                pool = [h.id for h in search(index, jds[jd_id], k=len(g), mode=mode,
                                             restrict=g.resume_id.unique().tolist())]
                ndcg.append(ndcg_at_k(pool, gains, 10))
                per_jd.append({"mode": mode, "jd_id": jd_id, "ndcg@10": ndcg[-1],
                               "recall@50": rec[50][-1] if rel else None,
                               "n_good": len(rel), "pool": len(g),
                               "jd_start": jds[jd_id][:120]})
        results[mode] = {
            **{f"recall@{k}": float(np.mean(v)) for k, v in rec.items()},
            "mrr": float(np.mean(rr)),
            "pool_ndcg@10": float(np.mean(ndcg)),
        }
    results["random"] = random_baseline(fit, len(docs), np.random.default_rng(SEED))

    out = {
        "component": "C3 hybrid retriever",
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {"embed_model": config.EMBED_MODEL, "chunk_words": config.CHUNK_WORDS,
                   "chunk_overlap": config.CHUNK_OVERLAP, "rrf_k": config.RRF_K,
                   "query": "raw JD text"},
        "n": {"corpus_resumes": len(docs),
              "jds_recall": int(fit.groupby("jd_id").relevance.max().eq(2).sum()),
              "jds_ndcg": int(fit.groupby("jd_id").relevance.max().gt(0).sum())},
        "metrics": results,
    }
    res_dir = config.EVAL_DIR / "results"
    (res_dir / "retrieval.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(per_jd).to_csv(res_dir / "retrieval_per_jd.csv", index=False)
    print(pd.DataFrame(results).T.round(3).to_string())


if __name__ == "__main__":
    main()
