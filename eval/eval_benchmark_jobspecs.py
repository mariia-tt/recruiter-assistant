"""Check the committed benchmark JobSpecs (input of the C3/C4 evals) against the JobSpec gold.

eval/results/benchmark_jobspecs.json was extracted with prompt jobspec_extractor v1; the
C1 eval measures the current prompt (v2). This script scores the stored v1 output on the
15 gold JDs with the same slot metrics, so the quality of the C3/C4 input is known.
0 LLM calls. Usage: uv run python eval/eval_benchmark_jobspecs.py
"""

import json
from datetime import UTC, datetime

from eval_collector import GOLD, aggregate

from recruiter import config

OUT = config.EVAL_DIR / "results" / "benchmark_jobspecs_quality.json"


def main() -> None:
    gold = {g["jd_id"]: g for g in map(json.loads, (GOLD / "jobspec_gold.jsonl").read_text().splitlines())}
    specs = json.loads(config.BENCHMARK_JOBSPECS.read_text())
    if missing := gold.keys() - specs.keys():
        raise SystemExit(f"{len(missing)} gold JDs have no stored JobSpec")
    out = {
        "component": "benchmark JobSpecs (jobspec_extractor v1 on full JDs)",
        "timestamp": datetime.now(UTC).isoformat(),
        "n": len(gold),
        "gold": "eval/gold/jobspec_gold.jsonl",
        "metrics": aggregate([(specs[k], g) for k, g in gold.items()]),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["metrics"], indent=1))


if __name__ == "__main__":
    main()
