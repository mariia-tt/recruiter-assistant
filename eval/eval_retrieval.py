"""Evaluate C3 retriever on the resume–JD fit benchmark (no LLM calls).

Two views:
- corpus retrieval: search all benchmark resumes with the JD; relevant = "Good Fit"
  resumes of that JD -> Recall@k, MRR (JDs with >= 1 Good Fit);
- pool ranking: rerank only the labelled resumes of each JD -> NDCG@10 with graded
  labels (0/1/2), over JDs with >= 1 non-zero label. Baseline for the C4 ranker.
Queries: raw JD text vs a query built from the LLM-extracted JobSpec
(eval/results/benchmark_jobspecs.json, made by scripts/extract_benchmark_jobspecs.py).
"""

import json
from collections.abc import Mapping
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from recruiter import config
from recruiter.metrics import ndcg_at_k, paired_bootstrap, recall_at_k, reciprocal_rank
from recruiter.retrieval import build_index, query_from_spec, search
from recruiter.schemas import JobSpec

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


def significance(per_jd: pd.DataFrame) -> dict:
    """Paired bootstrap of the chosen method against the raw-JD baselines."""
    chosen, out = "hybrid/jobspec", {}
    if chosen not in set(per_jd.method):
        return out
    for metric in ("recall@50", "ndcg@10"):
        wide = per_jd.pivot(index="jd_id", columns="method", values=metric)
        for base in ("hybrid/raw_jd", "bm25/raw_jd"):
            pair = wide[[chosen, base]].dropna()
            out[f"{metric}: {chosen} vs {base}"] = paired_bootstrap(
                pair[chosen].tolist(), pair[base].tolist()
            )
    return out


def main() -> None:
    fit = pd.read_parquet(config.DATA_PROCESSED / "fit_test.parquet")
    docs = fit.drop_duplicates("resume_id")
    index = build_index(docs.resume_id.tolist(), docs.resume_text.tolist())
    jds = fit.drop_duplicates("jd_id").set_index("jd_id").jd_text

    queries: dict[str, Mapping[str, str]] = {"raw_jd": jds}
    specs_file = config.BENCHMARK_JOBSPECS
    if not specs_file.exists():
        raise SystemExit(f"{specs_file} is missing (it is committed; restore it with git, or "
                         "regenerate with scripts/extract_benchmark_jobspecs.py — 71 LLM calls)")
    specs = json.loads(specs_file.read_text())
    queries["jobspec"] = {k: query_from_spec(JobSpec(**v)) for k, v in specs.items()}
    # Evaluate both query types on the same JDs so numbers are comparable.
    fit = fit[fit.jd_id.isin(specs)]

    results, per_jd = {}, []
    for qname, qtext in queries.items():
        for mode in MODES:
            name = f"{mode}/{qname}"
            rec = {k: [] for k in KS}
            rr, ndcg = [], []
            for jd_id, g in fit.groupby("jd_id"):
                rel = set(g.resume_id[g.relevance == 2])
                gains = dict(zip(g.resume_id, g.relevance, strict=True))
                if rel:
                    ranked = [h.id for h in search(index, qtext[jd_id], k=max(KS), mode=mode)]
                    for k in KS:
                        rec[k].append(recall_at_k(ranked, rel, k))
                    rr.append(reciprocal_rank(ranked, rel))
                if g.relevance.max() > 0:
                    pool = [h.id for h in search(index, qtext[jd_id], k=len(g), mode=mode,
                                                 restrict=g.resume_id.unique().tolist())]
                    ndcg.append(ndcg_at_k(pool, gains, 10))
                    per_jd.append({"method": name, "jd_id": jd_id, "ndcg@10": ndcg[-1],
                                   "recall@50": rec[50][-1] if rel else None,
                                   "n_good": len(rel), "pool": len(g),
                                   "jd_start": jds[jd_id][:120]})
            results[name] = {
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
                   "queries": list(queries)},
        "n": {"corpus_resumes": len(docs),
              "jds_recall": int(fit.groupby("jd_id").relevance.max().eq(2).sum()),
              "jds_ndcg": int(fit.groupby("jd_id").relevance.max().gt(0).sum())},
        "metrics": results,
        "significance": significance(pd.DataFrame(per_jd)),
    }
    res_dir = config.EVAL_DIR / "results"
    (res_dir / "retrieval.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(per_jd).to_csv(res_dir / "retrieval_per_jd.csv", index=False)
    print(pd.DataFrame(results).T.round(3).to_string())
    for name, r in out["significance"].items():
        print(f"{name}: {r['mean_diff']:+.3f} CI {[round(x, 3) for x in r['ci95']]}")


if __name__ == "__main__":
    main()
