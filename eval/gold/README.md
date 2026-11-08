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

## Vacancy requirements gold (C1 / JobSpec extraction)
- **Files:** `jobspec_sample.jsonl` (15 benchmark JDs, random seed 7), `jobspec_gold.jsonl`
  (labels), `jobspec_guideline.md`.
- **Annotator:** independent LLM agent (Claude), no access to the system's code, prompts or
  outputs; every `min_years` / `seniority` value has a verbatim evidence fragment.
- **Ambiguities** (recorded per JD in `notes`): JDs without a title (title inferred), JDs
  whose requirements are all "preferred" (empty must-haves), unmapped levels ("Staff",
  "V") resolved by years.

## Benchmark JobSpecs — LLM-generated eval input (C3, C4, C4b, C5)
- **File:** `eval/results/benchmark_jobspecs.json` — not a gold set, but data made by an LLM
  that the evals depend on, so it is committed and described here.
- **How it was made:** `scripts/extract_benchmark_jobspecs.py` ran the system's own JobSpec
  extractor on each of the 71 benchmark JDs (full JD text): prompt
  `prompts/jobspec_extractor.md` **v1**, `gemini-3.5-flash-lite`, temperature 0, 71 calls
  (2026-11-07). Re-running the script skips existing entries; `--force` re-extracts.
- **Read by:** `eval_retrieval.py` (structured queries), `eval_ranker.py`,
  `eval_reranker_llm.py`, `eval_outreach.py`.
- **Quality:** scored against `jobspec_gold.jsonl` (15 of these JDs) by
  `eval/eval_benchmark_jobspecs.py` → `eval/results/benchmark_jobspecs_quality.json`:
  title 0.93, seniority 0.87, min years 0.93, must-have F1 0.65 lexical / 0.70 semantic.
  Prompt v2 on the same JDs (C1 eval, "full JD" column): title 1.00, seniority 0.93,
  min years 1.00, must-have F1 0.64 / 0.71, nice-to-have F1 0.71. So must-haves are on par; v1 is 1 JD of 15 worse on title, seniority and min years (within noise at n = 15, but min years is exactly what v2 fixed) and 0.10 lower on nice-to-have F1.
  Not re-extracted with v2: ≈ 200 LLM calls (71 + invalidated C4b and C5 caches).

## Outreach claims gold (validates the C6 grounding checker)
- **Files:** `claims_sample.jsonl` (61 candidate claims extracted by C6 from the 20
  outreach messages of `eval/eval_outreach.py`, all attempts), `claims_gold.jsonl`
  (supported true/false + verbatim evidence), `claims_guideline.md`.
- **Annotator:** independent LLM agent (Claude) that did not see C6's verdicts.
- **Privacy:** the CV texts given to the annotator are not stored here (one source CV
  contains a real name the dataset did not anonymise); they are loaded by `cv_id`
  from `data/processed/resumes.parquet`, which is not committed.
- **Limitation:** validates C6's verdicts on the claims C6 itself extracted, not whether
  C6 extracts every claim; the latter is covered by the fabrication-injection test in
  `eval/eval_outreach.py` (20/20 injected fabrications caught).
