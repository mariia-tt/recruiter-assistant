# Gold / benchmark sets

## Retrieval & ranking benchmark (C3, C4)
- **Source:** `cnamuangtoun/resume-job-description-fit`, test split (open dataset on
  Hugging Face), cleaned by `scripts/prepare_data.py` → `data/processed/fit_test.parquet`.
- **Size:** 1 759 labelled resume–JD pairs; 476 unique resumes after cleaning
  (477 raw; two near-identical CVs merged by text hash); 71 JDs
  (28 with ≥ 1 "Good Fit", 61 with ≥ 1 non-zero label, 10 with only "No Fit").
- **Labels:** human-provided by the dataset authors: No Fit / Potential Fit /
  Good Fit, mapped to graded relevance 0 / 1 / 2.
- **Known limitation:** labels are sparse — each resume is labelled for only
  ~3.7 of the 71 JDs on average. Any resume without a label for a JD is treated
  as non-relevant, so corpus-level Recall@k is a *lower bound*. The pool-level
  NDCG (ranking only the labelled resumes of each JD) does not have this bias.
- Not copied into this folder: it is downloaded by `scripts/download_data.py`.
