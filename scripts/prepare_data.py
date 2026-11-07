"""Phase 3: clean raw datasets into data/processed/ and evaluate cleaning (C0)."""

import json
import re

import pandas as pd

from recruiter import config
from recruiter.data.clean import (
    EMAIL_RE,
    GLUED_BEFORE_RE,
    GLUED_SECTION_RE,
    PHONE_RE,
    clean_fit_benchmark,
    clean_resumes,
)


def issue_counts(texts: pd.Series) -> dict:
    """Number of documents still showing each noise pattern (C0 quality metric)."""
    return {
        "emails": int(texts.str.contains(EMAIL_RE).sum()),
        "phones": int(texts.str.contains(PHONE_RE).sum()),
        "glued_section_or_month": int(
            texts.map(lambda t: bool(GLUED_SECTION_RE.search(t) or GLUED_BEFORE_RE.search(t))).sum()
        ),
        "multi_space": int(texts.str.contains(re.compile(r" {2,}")).sum()),
        "exact_duplicates": int(texts.duplicated().sum()),
        "under_min_words": int((texts.str.split().str.len() < config.MIN_CV_WORDS).sum()),
    }


def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    raw_cv = pd.read_parquet(config.DATA_RAW / "resumes.parquet")
    cv, report = clean_resumes(raw_cv)
    cv.to_parquet(config.DATA_PROCESSED / "resumes.parquet", index=False)

    raw_fit = pd.read_parquet(config.DATA_RAW / "fit_test.parquet")
    fit = clean_fit_benchmark(raw_fit)
    fit.to_parquet(config.DATA_PROCESSED / "fit_test.parquet", index=False)

    result = {
        "candidate_db": {**report, "before": issue_counts(raw_cv.Resume_str), "after": issue_counts(cv.text)},
        "benchmark_resumes": {
            "before": issue_counts(raw_fit.resume_text.drop_duplicates()),
            "after": issue_counts(fit.resume_text.drop_duplicates()),
        },
    }
    out = config.EVAL_DIR / "results" / "data_cleaning.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
