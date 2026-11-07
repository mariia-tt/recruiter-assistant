"""Print aggregate statistics of the raw datasets for Data Understanding (Phase 2)."""

import re
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def length_stats(s: pd.Series) -> dict:
    """Character and word length percentiles of a text column."""
    chars, words = s.str.len(), s.str.split().str.len()
    q = [0, 0.5, 0.95, 1]
    return {
        "chars": chars.quantile(q).round().astype(int).tolist(),
        "words": words.quantile(q).round().astype(int).tolist(),
    }


def text_issues(s: pd.Series) -> dict:
    """Count common noise patterns that the cleaning step must handle."""
    return {
        "empty": int((s.str.strip().str.len() == 0).sum()),
        "exact_duplicates": int(s.duplicated().sum()),
        "html_tags": int(s.str.contains(r"<[a-z]+[^>]*>", flags=re.IGNORECASE).sum()),
        "emails": int(s.str.contains(r"[\w.]+@[\w.]+\.\w+").sum()),
        "phones": int(s.str.contains(r"\(?\d{3}\)?[ -]\d{3}-\d{4}").sum()),
        "company_name_placeholder": int(s.str.contains("Company Name").sum()),
        "glued_words": int(s.str.contains(r"[a-z][A-Z][a-z]").sum()),
    }


def explore_resumes() -> None:
    df = pd.read_parquet(RAW / "resumes.parquet")
    print("== resumes", df.shape, list(df.columns))
    print("missing:", df.isna().sum().to_dict())
    print("categories:", df.Category.nunique())
    print(df.Category.value_counts().to_string())
    print("length:", length_stats(df.Resume_str))
    print("issues:", text_issues(df.Resume_str))
    print("sample:", df.Resume_str.iloc[0][:300].replace("\n", " "))


def explore_fit() -> None:
    df = pd.read_parquet(RAW / "fit_test.parquet")
    print("\n== fit_test", df.shape, list(df.columns))
    print("labels:", df.label.value_counts().to_dict())
    per_jd = df.groupby("job_description_text").label.value_counts().unstack(fill_value=0)
    print("JDs:", len(per_jd), "| JDs with >=1 Good Fit:", int((per_jd["Good Fit"] > 0).sum()))
    print("pool size per JD p50/max:", int(per_jd.sum(axis=1).median()), int(per_jd.sum(axis=1).max()))
    print("unique resumes:", df.resume_text.nunique())
    print("resume length:", length_stats(df.resume_text))
    print("JD length:", length_stats(df.job_description_text))
    print("resume issues:", text_issues(df.resume_text))


if __name__ == "__main__":
    explore_resumes()
    explore_fit()
