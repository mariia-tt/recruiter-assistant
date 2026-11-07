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

## CV profiles — silver standard (C2a, C2b)
- **Files:** `cv_profiles_sample.jsonl` (input: 30 cleaned CVs from the candidate DB,
  15 randomly chosen categories × 2 CVs, seed 42), `cv_profiles.jsonl` (labels),
  `cv_profiles_guideline.md` (annotation rules), `skill_vocabulary.txt`.
- **Annotator:** an independent LLM agent (Claude) — a different model family from
  the system's Gemini model — instructed not to open the project's code or prompts.
  It read every CV fully; a helper script only merged the date ranges it entered.
- **Checks:** 30 valid records; all skills in vocabulary; every evidence fragment is a
  verbatim substring of its CV; every years value lists the ranges used (auditable).
- **Known ambiguity:** the guideline did not define precedence when a headline holds
  two seniority keywords ("Assistant Manager"); the annotator applied list order.
- **Status:** silver, not human-verified. A human spot-check of a few records is
  welcome but was not required for the reported numbers.
