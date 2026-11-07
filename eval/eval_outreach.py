"""Evaluate C5 outreach writer + C6 grounding checker.

Pairs: 10 benchmark JobSpecs (seed 42) x top-2 candidates from the candidate DB = 20 messages.
Per pair: C2b profile (1 call) + write/check loop (2-4 calls). Then a perturbation test of C6:
a fabricated claim is inserted into each final message and C6 must flag it (1 call each).
Results are saved after every pair (resumable; LLM responses are cached).
Usage: uv run python eval/eval_outreach.py --limit 2
"""

import argparse
import json
import random
from datetime import UTC, datetime

import numpy as np

from recruiter import config, llm
from recruiter.outreach import CHECKER, WRITER, check_grounding, generate_outreach
from recruiter.pipeline import load_db, search_candidates
from recruiter.profiler_llm import profile_cv
from recruiter.schemas import JobSpec

SEED, N_JDS, PER_JD = 42, 10, 2
OUT = config.EVAL_DIR / "results" / "outreach_messages.jsonl"
FABRICATIONS = [
    ("You also spent seven years as a commercial airline pilot.", ["pilot", "airline"]),
    ("Your PhD in marine biology also stands out.", ["phd", "marine"]),
    ("Leading a 40-person team at NASA is impressive.", ["nasa", "40"]),
    ("Your work as a professional chef in Paris is unique.", ["chef", "paris"]),
]


def pairs() -> list[tuple[str, JobSpec, str]]:
    specs = json.loads((config.DATA_PROCESSED / "benchmark_jobspecs.json").read_text())
    jd_ids = random.Random(SEED).sample(sorted(specs), N_JDS)
    out = []
    for jd_id in jd_ids:
        spec = JobSpec(**specs[jd_id])
        for c in search_candidates(spec)[:PER_JD]:
            out.append((jd_id, spec, c))
    return out


def run(limit: int | None) -> list[dict]:
    done = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done[(r["jd_id"], r["cv_id"])] = r
    db = load_db()
    rows = []
    for jd_id, spec, score in pairs()[:limit]:
        key = (jd_id, score.id)
        if key in done:
            rows.append(done[key])
            continue
        text = db.texts[score.id]
        try:
            prof = profile_cv(text)
            res = generate_outreach(spec, prof, text, score)
            fab, keys = FABRICATIONS[len(rows) % len(FABRICATIONS)]
            perturbed = res.message.text.rstrip() + " " + fab
            rep = check_grounding(perturbed, text)
        except llm.QuotaExhaustedError as e:
            print(f"STOP: {e}")
            break
        caught = any(not c.supported and any(k in c.claim.lower() for k in keys)
                     for c in rep.claims)
        rows.append({"jd_id": jd_id, "cv_id": score.id, "title": spec.title,
                     "result": res.model_dump(), "fabrication": fab, "fabrication_caught": caught})
        with open(OUT, "a") as f:
            f.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
    return rows


def metrics(rows: list[dict]) -> dict:
    first = [r["result"]["history"][0] for r in rows]
    final = [r["result"] for r in rows]

    def faith(hs: list[dict]) -> float:
        claims = sum(h["n_claims"] for h in hs)
        return 1 - sum(len(h["unsupported"]) for h in hs) / claims if claims else 0.0

    final_hist = [r["history"][-1] for r in final]
    supported_counts = [sum(c["supported"] for c in r["report"]["claims"]) for r in final]
    return {
        "n_messages": len(rows),
        "first_draft": {
            "faithfulness_claim_level": faith(first),
            "messages_with_unsupported_claim": float(np.mean([bool(h["unsupported"]) for h in first])),
            "constraint_compliance": float(np.mean([not h["violations"] for h in first])),
        },
        "final": {
            "faithfulness_claim_level": faith(final_hist),
            "messages_with_unsupported_claim": float(np.mean([bool(h["unsupported"]) for h in final_hist])),
            "constraint_compliance": float(np.mean([not r["violations"] for r in final])),
            "passed_all_checks": float(np.mean([r["passed"] for r in final])),
            "personalised_>=2_supported_facts": float(np.mean([n >= 2 for n in supported_counts])),
            "mean_supported_facts": float(np.mean(supported_counts)),
            "mean_words": float(np.mean([len(r["message"]["text"].split()) for r in final])),
        },
        "mean_attempts": float(np.mean([r["attempts"] for r in final])),
        "c6_fabrication_detection_rate": float(np.mean([r["fabrication_caught"] for r in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    rows = run(args.limit)
    out = {
        "component": "C5 outreach writer + C6 grounding checker",
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {"model": config.LLM_MODEL, "writer_prompt": llm.prompt_version(WRITER),
                   "checker_prompt": llm.prompt_version(CHECKER),
                   "max_words": config.OUTREACH_MAX_WORDS},
        "metrics": metrics(rows) if rows else {},
        "llm_calls_used_today": llm.usage_today(config.LLM_MODEL),
    }
    (config.EVAL_DIR / "results" / "outreach.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["metrics"], indent=1), "\nused today:", out["llm_calls_used_today"])


if __name__ == "__main__":
    main()
