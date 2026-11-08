"""Extract a JobSpec for every benchmark JD (1 LLM call per JD, cached).

The output file is committed; JDs that already have a spec are skipped, so a run on a
fresh clone makes 0 LLM calls. Use --force to re-extract (e.g. after a prompt change).
Usage: uv run python scripts/extract_benchmark_jobspecs.py [--limit 3] [--force]
"""

import argparse
import json

import pandas as pd

from recruiter import config, llm
from recruiter.jobspec import extract_jobspec

OUT = config.BENCHMARK_JOBSPECS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="process only the first N JDs")
    parser.add_argument("--force", action="store_true",
                        help="re-extract JDs that already have a spec (costs LLM calls)")
    args = parser.parse_args()

    fit = pd.read_parquet(config.DATA_PROCESSED / "fit_test.parquet")
    jds = fit.drop_duplicates("jd_id").sort_values("jd_id")[["jd_id", "jd_text"]]
    if args.limit:
        jds = jds.head(args.limit)

    specs = json.loads(OUT.read_text()) if OUT.exists() else {}
    for row in jds.itertuples():
        if row.jd_id in specs and not args.force:
            continue
        try:
            specs[row.jd_id] = extract_jobspec(row.jd_text).model_dump()
        except llm.QuotaExhaustedError as e:
            print(f"STOP: {e}")
            break
        OUT.write_text(json.dumps(specs, indent=1))
    print(f"{len(specs)} JobSpecs in {OUT.name}; used today: {llm.usage_today(config.LLM_MODEL)}")


if __name__ == "__main__":
    main()
