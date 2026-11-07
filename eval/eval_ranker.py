"""Evaluate C4 scorecard ranker on labelled pools of the fit benchmark (no LLM calls).

For each JD with >= 1 non-zero label the labelled resumes are re-ranked and compared
with graded labels (Good=2, Potential=1, No=0). Reports:
- NDCG@10, P@5 (Good Fit), Spearman(total, label) for retrieval baselines and scorecard variants;
- ablation: drop one criterion at a time (weights renormalised);
- weight tuning with 2-fold cross-validation over JDs (to check business weights);
- knock-out rule quality: how many Good Fit candidates it wrongly rejects.
"""

import itertools
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from recruiter import config
from recruiter.metrics import ndcg_at_k, paired_bootstrap, precision_at_k
from recruiter.profiler_rules import profile
from recruiter.ranker import score_candidate, similarity_features
from recruiter.retrieval import build_index, query_from_spec, search
from recruiter.schemas import JobSpec

SEED = 42
CRITERIA = list(config.SCORECARD_WEIGHTS)


def build_features(fit: pd.DataFrame, specs: dict, index) -> pd.DataFrame:
    """One row per (JD, labelled resume): criterion scores, knock-out flag, retrieval scores."""
    texts = dict(zip(fit.resume_id, fit.resume_text, strict=False))
    profiles = {rid: profile(rid, t) for rid, t in texts.items()}
    rows = []
    for jd_id, g in fit.groupby("jd_id"):
        if g.relevance.max() == 0:
            continue
        spec = JobSpec(**specs[jd_id])
        pool = [profiles[r] for r in g.resume_id]
        title_sim, domain_sim, semantic_sim = similarity_features(spec, pool, index)
        query = query_from_spec(spec)
        hybrid_rank = {h.id: i for i, h in enumerate(
            search(index, query, k=len(g), restrict=g.resume_id.tolist()))}
        for p, rel in zip(pool, g.relevance, strict=True):
            c = score_candidate(spec, p, texts[p.id], title_sim[p.id], domain_sim[p.id],
                                semantic_sim[p.id], knockout=0.5)
            row = {"jd_id": jd_id, "id": p.id, "relevance": rel, "knocked_out": c.knocked_out,
                   "hybrid": -hybrid_rank[p.id]}
            row |= {x.name: x.score for x in c.criteria}
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, score: pd.Series, knockout: bool = False) -> dict:
    """Mean ranking metrics over JDs for a given candidate score."""
    ndcg, p5, rho, per_jd = [], [], [], {}
    for jd_id, g in df.assign(_s=score).groupby("jd_id"):
        key = [g.knocked_out, -g._s] if knockout else [-g._s]
        order = g.iloc[np.lexsort(key[::-1])].id.tolist()
        gains = dict(zip(g.id, g.relevance, strict=True))
        ndcg.append(ndcg_at_k(order, gains, 10))
        p5.append(precision_at_k(order, set(g.id[g.relevance == 2]), 5))
        if g._s.nunique() > 1 and g.relevance.nunique() > 1:
            rho.append(spearmanr(g._s, g.relevance).statistic)
        per_jd[jd_id] = ndcg[-1]
    return {"ndcg@10": float(np.mean(ndcg)), "p@5": float(np.mean(p5)),
            "spearman": float(np.mean(rho)), "_per_jd": per_jd}


def weighted(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    total = sum(weights.values()) or 1
    return sum(df[c] * w for c, w in weights.items()) / total


def weight_grid(features: list[str], step: float = 0.1):
    """All weight vectors on a simplex grid (non-negative, sum to 1)."""
    n = round(1 / step)
    for cuts in itertools.combinations(range(n + len(features) - 1), len(features) - 1):
        parts = np.diff([-1, *cuts, n + len(features) - 1]) - 1
        yield dict(zip(features, parts * step, strict=True))


def tune_cv(df: pd.DataFrame, features: list[str]) -> dict:
    """2-fold CV over JDs: pick best weights on one half, report NDCG on the other."""
    jds = np.array(sorted(df.jd_id.unique()))
    np.random.default_rng(SEED).shuffle(jds)
    folds = [set(jds[: len(jds) // 2]), set(jds[len(jds) // 2 :])]
    grid = list(weight_grid(features))
    out = {"held_out_ndcg": [], "business_ndcg_same_folds": [], "best_weights": []}
    for train, test in [(folds[0], folds[1]), (folds[1], folds[0])]:
        tr, te = df[df.jd_id.isin(train)], df[df.jd_id.isin(test)]
        best = max(grid, key=lambda w: evaluate(tr, weighted(tr, w))["ndcg@10"])
        out["best_weights"].append({k: round(float(v), 2) for k, v in best.items()})
        out["held_out_ndcg"].append(evaluate(te, weighted(te, best))["ndcg@10"])
        out["business_ndcg_same_folds"].append(
            evaluate(te, weighted(te, config.SCORECARD_WEIGHTS_V1))["ndcg@10"])
        out.setdefault("v2_ndcg_same_folds", []).append(
            evaluate(te, weighted(te, config.SCORECARD_WEIGHTS))["ndcg@10"])
    out["mean_held_out_ndcg"] = float(np.mean(out["held_out_ndcg"]))
    out["mean_business_ndcg"] = float(np.mean(out["business_ndcg_same_folds"]))
    out["mean_v2_ndcg"] = float(np.mean(out["v2_ndcg_same_folds"]))
    return out


def knockout_report(df: pd.DataFrame) -> dict:
    """How the must-have knock-out rule treats each label class."""
    by = df.groupby("relevance").knocked_out.mean()
    return {"knocked_out_share_by_label": {str(k): round(float(v), 3) for k, v in by.items()},
            "good_fit_wrongly_rejected": int(df[(df.relevance == 2) & df.knocked_out].shape[0]),
            "good_fit_total": int((df.relevance == 2).sum())}


def main() -> None:
    fit = pd.read_parquet(config.DATA_PROCESSED / "fit_test.parquet")
    specs = json.loads((config.DATA_PROCESSED / "benchmark_jobspecs.json").read_text())
    docs = fit.drop_duplicates("resume_id")
    index = build_index(docs.resume_id.tolist(), docs.resume_text.tolist())
    df = build_features(fit, specs, index)
    df.to_csv(config.EVAL_DIR / "results" / "ranker_features.csv", index=False)

    v1, v2 = config.SCORECARD_WEIGHTS_V1, config.SCORECARD_WEIGHTS
    methods = {
        "random": (pd.Series(np.random.default_rng(SEED).random(len(df)), index=df.index), False),
        "retrieval: hybrid/jobspec (C3)": (df.hybrid, False),
        "retrieval: dense/jobspec": (df.semantic, False),
        "scorecard v1 (business weights)": (weighted(df, v1), False),
        "scorecard v1 + knock-out 0.5": (weighted(df, v1), True),
        "scorecard v2 (revised weights)": (weighted(df, v2), False),
    }
    for c in CRITERIA:
        methods[f"ablation v2: without {c}"] = (
            weighted(df, {k: v for k, v in v2.items() if k != c}), False)
    results = {name: evaluate(df, s, ko) for name, (s, ko) in methods.items()}

    base = "retrieval: hybrid/jobspec (C3)"
    jd_ids = list(results[base]["_per_jd"])
    significance = {
        f"{m} vs {base}": paired_bootstrap([results[m]["_per_jd"][j] for j in jd_ids],
                                           [results[base]["_per_jd"][j] for j in jd_ids])
        for m in ("scorecard v1 (business weights)", "scorecard v2 (revised weights)")
    }
    out = {
        "component": "C4 scorecard ranker",
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {"weights_v1": v1, "weights_v2": v2, "knockout_variant": 0.5,
                   "sim_range": [config.SIM_LOW, config.SIM_HIGH]},
        "n": {"jds": int(df.jd_id.nunique()), "candidates": len(df)},
        "metrics": {k: {m: v for m, v in r.items() if m != "_per_jd"} for k, r in results.items()},
        "significance": significance,
        "knockout": knockout_report(df),
        "tuning_cv": {"all_criteria": tune_cv(df, CRITERIA)},
    }
    (config.EVAL_DIR / "results" / "ranker.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(out["metrics"]).T.round(3).to_string())
    for k, v in significance.items():
        print(f"{k}: {v['mean_diff']:+.3f} CI {[round(x, 3) for x in v['ci95']]}")
    print("knock-out:", out["knockout"])
    for k, v in out["tuning_cv"].items():
        print(f"tuning {k}: tuned held-out {v['mean_held_out_ndcg']:.3f} | v1 "
              f"{v['mean_business_ndcg']:.3f} | v2 {v['mean_v2_ndcg']:.3f}; "
              f"best {v['best_weights']}")


if __name__ == "__main__":
    main()
