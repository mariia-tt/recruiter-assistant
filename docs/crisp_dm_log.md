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

### 2026-11-07 — Phase 5 (loop-back to 4): C2a/C2b profiler evaluation
- **What was done:** 30-CV silver standard built by an independent LLM annotator
  (Claude subagent, no access to code/prompts); C2b LLM profiler (prompt
  `cv_profiler` v1); `eval/eval_profiler.py`; C4/C4b evals re-run after C2a fix.
- **Decision & why:** C2a fixed (unicode dashes, education dates excluded).
  Ranking keeps rule profiles; C2b only for shortlisted CVs (achievements, roles)
  because rules beat the LLM on skills recall, years and seniority.
- **Evidence:** C2a years MAE 1.69 → 0.39, skills F1 0.71, seniority 0.90;
  C2b skills F1 0.52 (P 0.86), years MAE 1.32, achievements supported 100 %.
  DB-wide: 1 205 CVs use the unparsed dashes; 0-year CVs 2.5 % → 1.3 %.
  C4 after fix: v2 NDCG@10 0.808 (vs C3 +0.060, CI [+0.012, +0.107]); CV 0.800.
  30 LLM calls.
- **Loop-back:** Modeling (C2a parser), then Evaluation of C4/C4b re-run.
- **Next step:** Streamlit app (pipeline already wired), C1 dialog, C5/C6.

### 2026-11-07 — Phase 6: Deployment — Streamlit app and C1 collector
- **What was done:** C1 requirements collector (`collector.py`): LLM re-extracts
  the JobSpec from the whole chat each turn, rule-based follow-up questions
  (max 2 per turn, search after 4 turns or on "search"); `app.py` with chat,
  JobSpec view, candidate cards (score breakdown, matched/missing must-haves,
  CV evidence, optional LLM assessment), LLM budget counter; README.
- **Decision & why:** follow-up policy is deterministic — predictable, testable,
  and 1 LLM call per recruiter message on the free tier.
- **Evidence:** 33 unit tests; end-to-end AppTest smoke run with a fake LLM
  (dialog → follow-up → search → cards); Streamlit server health check ok.
- **Loop-back:** none
- **Next step:** C5 outreach writer + C6 grounding checker; C1 evaluation on
  simulated dialogs.

### 2026-11-07 — Phase 4/5: C5/C6 outreach, C1 evaluation, judge validation
- **What was done:** C5 writer + C6 grounding checker with rewrite loop
  (`outreach.py`), outreach button in the app; `eval_outreach.py` (20 messages +
  fabrication injection), `eval_collector.py` (15 simulated dialogs vs independent
  gold), `eval_grounding_judge.py` (C6 vs independent claim labels, 61 claims).
- **Decision & why:** (1) extractor prompt v2 — never derive min_years from level
  (it invented "5 years" and suppressed the question); (2) collector asks once for
  more skills when < 3 must-haves and never re-asks a slot; (3) writer no longer
  gets computed total years (unverifiable; C2b years MAE 1.3 y).
- **Evidence:** C1 dialog: title 1.00, seniority 1.00, years 0.93, must-have F1
  0.48 lexical / 0.52 semantic (upper bound 0.64 / 0.71), resolved 0.93, 2.3 turns.
  C5 v1 → v2: first-draft messages with unsupported claims 20 % → 10 %, final
  faithfulness 0.965 → 1.000, personalised 95 %. C6: accuracy 0.95, kappa 0.74,
  20/20 injected fabrications caught. ~240 LLM calls (409/450 used today).
- **Loop-back:** Modeling (prompt v2, collector rules, writer facts) after
  evaluation; C1 completion metric redefined as "resolved" (2 of 3 "incomplete"
  dialogs were correct — JD has no years/level).
- **Next step:** final report with diagrams, video plan.

### 2026-11-07 — Phase 6: Deployment — final report and video plan
- **What was done:** `report.ipynb` / `report.html` built by `scripts/build_report.py`
  (all numbers read from `eval/results`), SVG data-flow diagram
  (`scripts/make_diagram.py`), trace viewer `scripts/show_traces.py`, video plan in
  Ukrainian (kept locally, not in the repository).
- **Decision & why:** notebook report so every table is regenerated from the eval
  outputs; ranking chart drawn as a dot plot (axis does not start at 0, bars would
  exaggerate differences).
- **Evidence:** notebook executes with 0 errors; 39 tests pass.
- **Loop-back:** none
- **Next step:** live run of the app with the real LLM before recording (quota reset).

### 2026-11-08 — Phase 5/6: Evaluation & Deployment — reproducible evals on a fresh clone
- **What was done:** cloned the repo into an empty folder and followed the README
  (`uv sync`, ruff, pytest, `download_data.py`, `prepare_data.py`, first search, the
  0-LLM evals). Found that `benchmark_jobspecs.json` (input of the C3/C4/C4b/C5 evals)
  lived only in uncommitted `data/processed/`: on a fresh clone `eval_ranker`,
  `eval_reranker_llm` and `eval_outreach` failed with FileNotFoundError and
  `eval_retrieval` silently dropped the JobSpec queries and overwrote `retrieval.json`
  with different numbers. Moved the file to `eval/results/` (committed, path in
  `config.BENCHMARK_JOBSPECS`); the extractor now skips JDs that already have a spec
  (`--force` to re-extract); `eval_retrieval` / `eval_ranker` fail with a clear message
  when the file is missing or incomplete; new `eval/eval_benchmark_jobspecs.py` scores
  the stored specs against `jobspec_gold.jsonl`; README and `eval/gold/README.md` updated.
- **Decision & why:** commit the LLM-generated eval input instead of regenerating it —
  regeneration costs 71 calls of the free daily quota and is not bit-for-bit
  reproducible (model drift), while the file is small (25 KB) and holds only fields
  extracted from public JDs, no candidate data.
- **Evidence:** on the fresh clone `data_cleaning.json`, `retrieval.json`, `ranker.json`
  and `grounding_judge.json` were reproduced exactly (only timestamps differ);
  39 tests pass; extractor re-run made 0 LLM calls. Stored v1 specs vs gold (v2 in
  brackets): must-have F1 0.65 / 0.70 (0.64 / 0.71), title 0.93 (1.00), seniority 0.87
  (0.93), min years 0.93 (1.00), nice-to-have F1 0.61 (0.71) — C3/C4 run on slightly
  noisier input; kept because re-extraction costs ≈ 200 calls. First app start
  on a fresh clone took ~15 min (model download + index), README now says 7–15 min.
- **Loop-back:** Deployment → Evaluation (eval inputs made reproducible).
- **Next step:** slides and video recording.
