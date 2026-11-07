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

### 2026-11-07 — Phase 4/5: Modeling + Evaluation — JobSpec queries for C3
- **What was done:** JobSpec extractor (`src/recruiter/jobspec.py`, prompt
  `jobspec_extractor` v1) — shared by C1; extracted JobSpecs for all 71
  benchmark JDs; retrieval eval now compares raw-JD vs JobSpec queries.
- **Decision & why:** default retrieval = hybrid with JobSpec query (best
  Recall@50 and MRR). Timeouts made retryable in `llm.py` after one run crashed
  on a 90 s timeout (44/71 saved, resumed from cache).
- **Evidence:** hybrid Recall@50 0.316 → 0.455 (+0.139, 95 % CI [+0.052,
  +0.248]); MRR 0.236 → 0.322; pool NDCG@10 0.700 → 0.748 (CI [+0.010, +0.088]).
  76 LLM calls used.
- **Loop-back:** corrected benchmark counts in docs: 61 (not 69) JDs have a
  non-zero label; corpus is 476 resumes after cleaning.
- **Next step:** C4 scorecard ranker — must beat pool NDCG@10 = 0.748.

### 2026-11-07 — Phase 4/5 (loop-back to Phase 1): C4 scorecard ranker + C4b LLM re-ranker
- **What was done:** `ranker.py` (weighted criteria with evidence snippets,
  optional knock-out), `reranker_llm.py` (prompt `llm_reranker` v1), evals with
  ablation, 2-fold CV weight tuning, knock-out analysis, paired bootstrap.
- **Decision & why:** (1) knock-out off by default — it rejected 87 % of Good Fit
  vs 90 % of No Fit; (2) weights v2 with a new `semantic` criterion and higher
  `experience` weight (ablation + CV tuning); (3) LLM re-ranker used for
  explanations only, not ordering; unverified quotes dropped.
- **Evidence:** NDCG@10 C3 0.748 → v1 0.754 (n.s.) → v2 0.814 (+0.065, CI
  [+0.017, +0.114]); unbiased CV estimate 0.786. LLM reorder 0.805 (−0.009, n.s.);
  LLM overall vs label Spearman 0.37; quotes verbatim 93 %. 61 + 2 LLM calls.
- **Loop-back:** to Phase 1 — business rules revised (knock-out, weights v2)
  in `docs/01_business_understanding.md`; Good Fit CVs contain only 19 % of
  JD must-haves literally, so labels reflect holistic role fit.
- **Next step:** C2 gold set (user verification), C1 dialog collector, C5/C6
  outreach writer + grounding checker, pipeline + Streamlit app.
