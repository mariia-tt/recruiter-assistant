"""Evaluate CV profilers C2a (rules) and C2b (LLM) on the silver-standard set.

Gold: eval/gold/cv_profiles.jsonl (30 CVs, independent LLM annotator, see eval/gold/README.md).
C2b makes 1 LLM call per CV (cached). Usage: uv run python eval/eval_profiler.py --limit 3
"""

import argparse
import json
import re
from datetime import UTC, datetime

import numpy as np

from recruiter import config, llm
from recruiter.profiler_llm import PROMPT, profile_cv
from recruiter.profiler_rules import SENIOR_WORDS, profile
from recruiter.skills import SKILL_ALIASES

GOLD = config.EVAL_DIR / "gold"


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def ambiguous_level(headline: str) -> bool:
    """Headline contains keywords of more than one seniority level (guideline gap)."""
    return sum(bool(re.search(p, headline.lower())) for p in SENIOR_WORDS.values()) > 1


def skill_prf(pred: list[set[str]], gold: list[set[str]]) -> dict:
    """Micro precision / recall / F1 over the shared skill vocabulary."""
    tp = sum(len(p & g) for p, g in zip(pred, gold, strict=True))
    fp = sum(len(p - g) for p, g in zip(pred, gold, strict=True))
    fn = sum(len(g - p) for p, g in zip(pred, gold, strict=True))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def supported(fact: str, cv: str, threshold: float = 0.7) -> bool:
    """Grounding proxy: >= 70 % of the fact's content words (len > 3) occur in the CV."""
    words = [w for w in _norm(fact).split() if len(w) > 3]
    cv_words = set(_norm(cv).split())
    return bool(words) and sum(w in cv_words for w in words) / len(words) >= threshold


def score(preds: list[dict], gold: list[dict]) -> dict:
    """All profile metrics for one system against the gold rows (same order)."""
    vocab = set(SKILL_ALIASES)
    years_err = [abs(p["years_experience"] - g["years_experience"]) for p, g in zip(preds, gold, strict=True)]
    level_ok = [p["seniority"] == g["seniority"] for p, g in zip(preds, gold, strict=True)]
    unamb = [ok for ok, g in zip(level_ok, gold, strict=True) if not ambiguous_level(g["headline"])]
    return {
        "headline_exact": float(np.mean([_norm(p["headline"]) == _norm(g["headline"])
                                         for p, g in zip(preds, gold, strict=True)])),
        "years_mae": float(np.mean(years_err)),
        "years_within_1y": float(np.mean([e <= 1 for e in years_err])),
        "seniority_accuracy": float(np.mean(level_ok)),
        "seniority_accuracy_unambiguous": float(np.mean(unamb)),
        "n_unambiguous": len(unamb),
        "skills": skill_prf([set(p["skills"]) & vocab for p in preds],
                            [set(g["skills"]) for g in gold]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="C2b: profile only first N CVs")
    args = parser.parse_args()

    sample = [json.loads(x) for x in (GOLD / "cv_profiles_sample.jsonl").read_text().splitlines()]
    gold = [json.loads(x) for x in (GOLD / "cv_profiles.jsonl").read_text().splitlines()]
    assert [s["id"] for s in sample] == [g["id"] for g in gold]

    rules = [profile(s["id"], s["text"]).model_dump() for s in sample]
    results = {"C2a rules": score(rules, gold)}

    n = args.limit or len(sample)
    llm_preds, stopped = [], None
    for s in sample[:n]:
        try:
            llm_preds.append(profile_cv(s["text"]).model_dump())
        except llm.QuotaExhaustedError as e:
            stopped = str(e)
            break
    if llm_preds:
        k = len(llm_preds)
        results["C2b LLM"] = score(llm_preds, gold[:k])
        results["C2a rules (same CVs)"] = score(rules[:k], gold[:k])
        facts = [(a, s["text"]) for p, s in zip(llm_preds, sample, strict=False) for a in p["achievements"]]
        results["C2b LLM"]["achievements_supported"] = float(np.mean([supported(a, t) for a, t in facts]))
        results["C2b LLM"]["n_achievements"] = len(facts)

    worst = sorted(zip(rules, gold, strict=True),
                   key=lambda pg: -abs(pg[0]["years_experience"] - pg[1]["years_experience"]))[:5]
    out = {
        "component": "C2a rule profiler / C2b LLM profiler",
        "timestamp": datetime.now(UTC).isoformat(),
        "gold": "silver standard, 30 CVs, independent LLM annotator (Claude)",
        "config": {"model": config.LLM_MODEL, "prompt_version": llm.prompt_version(PROMPT),
                   "c2b_n": len(llm_preds), "stopped": stopped},
        "metrics": results,
        "worst_years_c2a": [{"id": g["id"], "pred": p["years_experience"],
                             "gold": g["years_experience"]} for p, g in worst],
    }
    (config.EVAL_DIR / "results" / "profiler.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["metrics"], indent=1))
    print("worst years (C2a):", out["worst_years_c2a"])


if __name__ == "__main__":
    main()
