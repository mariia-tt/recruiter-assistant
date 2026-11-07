"""Download the open datasets used by the project into data/raw/ (Phase 2)."""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

SOURCES = {
    # Kaggle Resume Dataset mirror: candidate database for the demo app.
    "resumes.parquet": "hf://datasets/opensporks/resumes/Resume/Resume.csv",
    # Resume–JD pairs with fit labels: benchmark for retrieval and ranking.
    "fit_test.parquet": "hf://datasets/cnamuangtoun/resume-job-description-fit/test.csv",
}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        out = RAW / name
        if out.exists():
            print(f"skip {name} (exists)")
            continue
        df = pd.read_csv(url)
        df.to_parquet(out, index=False)
        print(f"saved {name}: {df.shape}")


if __name__ == "__main__":
    main()
