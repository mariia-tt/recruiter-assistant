# CRISP-DM log

### 2026-10-29 — Phase 1: Business Understanding — scope and success criteria
- **What was done:** chose the Recruiter Assistant task; defined scope, scorecard
  criteria, out-of-scope items and per-component success criteria
  (`docs/01_business_understanding.md`).
- **Decision & why:** recruiter task has open datasets with fit labels, so
  ranking can be evaluated on real labels instead of synthetic ones.
- **Evidence:** `cnamuangtoun/resume-job-description-fit` test split — 71 JDs,
  ~25 labelled resumes per JD (857 No / 444 Potential / 458 Good Fit).
- **Loop-back:** none
- **Next step:** Phase 2 — download and profile both datasets.

### 2026-10-29 — Phase 2: Data Understanding — profile both datasets
- **What was done:** downloaded candidate DB (2 484 CVs) and fit benchmark
  (1 759 labelled pairs); profiled size, labels, lengths, noise, PII
  (`docs/02_data_understanding.md`).
- **Decision & why:** use real open datasets instead of synthetic CVs; the
  benchmark provides human fit labels for ranking evaluation.
- **Evidence:** CVs are anonymised ("Company Name" in 2 483/2 484), 1 empty CV,
  PII in ~70 CVs, only 28/71 benchmark JDs have a "Good Fit" resume.
- **Loop-back:** Phase 1 updated implicitly — outreach messages cannot use
  candidate names (data is anonymised).
- **Next step:** Phase 3 — cleaning module with the rules from the findings table.

### 2026-10-29 — Phase 3: Data Preparation — cleaning module (C0)
- **What was done:** `src/recruiter/data/clean.py` + `scripts/prepare_data.py`:
  PII masking, glued-word splitting, whitespace normalisation, min-length
  filter, dedupe; benchmark gets resume/JD ids and graded relevance (0/1/2).
- **Decision & why:** LLM provider = Gemini (free tier, OpenAI-compatible
  endpoint). Free-tier limits → profiler split into rule-based C2a (whole DB)
  and LLM C2b (shortlist only, cached). Client-side throttle + disk cache in
  `llm.py`.
- **Evidence:** `eval/results/data_cleaning.json` — candidate DB 2 484 → 2 481
  rows; e-mails 19 → 0, phones 60 → 2 (both manually checked: licence numbers,
  false positives), glued words 27 → 0 (DB) / 476 → 0 (benchmark), duplicates
  2 → 0. Note: glued-word "after" count uses the same regex as the fix, so it
  is backed by unit tests (`tests/test_clean.py`) rather than by itself.
- **Loop-back:** none
- **Next step:** Phase 4 — schemas, C2a rule profiler, C3 retriever, C4 ranker.

### 2026-11-06 — Phase 4/5: Modeling + Evaluation — C2a rule profiler, C3 retriever
- **What was done:** schemas, skills dictionary, rule profiler C2a (skills,
  years from merged date ranges, seniority), hybrid retriever C3 (BM25 +
  bge-small chunks, RRF), `eval/eval_retrieval.py`.
- **Decision & why:** C2a sanity check on 2 481 CVs → fixed headline extraction
  (section prefixes) and seniority mapping ("manager" = senior, not lead).
  Long careers (> 30 y, 5.6 %) were checked manually and are genuine.
- **Evidence:** `eval/results/retrieval.json` — Recall@50: random 0.105, BM25
  0.231, dense 0.361, hybrid 0.316; MRR hybrid 0.236 (best); pool NDCG@10
  hybrid 0.700 vs random 0.574.
- **Loop-back:** Recall@50 below 0.80 target → (a) success criterion in Phase 1
  was set assuming dense labels; benchmark labels are sparse (43/50 top hits
  unlabelled), so corpus recall is a lower bound; (b) back to Modeling: build
  queries from LLM-extracted JobSpec instead of raw JD text.
- **Next step:** JD → JobSpec extraction (71 LLM calls, needs user approval),
  re-run C3 eval with structured queries.
