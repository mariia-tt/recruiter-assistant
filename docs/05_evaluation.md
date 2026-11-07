# Phase 5 — Evaluation

One section per component. Raw numbers live in `eval/results/*.json`.

## C0 Data cleaning
See `eval/results/data_cleaning.json` and the Phase 3 entry in
`docs/crisp_dm_log.md`: residual PII 79 → 2 documents (both manually verified
false positives — licence numbers), duplicates 2 → 0, glued words removed
(verified by unit tests).

## C2a rule profiler / C2b LLM profiler
**Script:** `eval/eval_profiler.py` → `eval/results/profiler.json` · **Gold:**
`eval/gold/cv_profiles.jsonl` — 30 CVs (15 categories × 2, seed 42), **silver
standard** annotated by an independent LLM annotator (Claude, a different model
family from the Gemini model under test) following
`eval/gold/cv_profiles_guideline.md`, without access to the system's code or
prompts. All 30 records passed automatic checks (every evidence fragment is a
verbatim substring of the CV; all skills from the vocabulary); every
`years_experience` value lists the date ranges used, so it can be audited.

**Why these metrics.** Profiles feed the ranker: skills → micro P/R/F1 over the
119-skill vocabulary; years → MAE and share within ±1 year (the experience
criterion is linear in years); seniority → accuracy; headline → exact match.

| Metric | C2a rules (all CVs, offline) | C2b LLM (`gemini-3.5-flash-lite`, prompt v1) | Target |
|---|---|---|---|
| Skills F1 (P / R) | **0.71** (0.64 / 0.78) | 0.52 (**0.86** / 0.37) | ≥ 0.75 |
| Years MAE | **0.39** y | 1.32 y | ≤ 1.5 y |
| Years within ±1 y | **97 %** | 60 % | |
| Seniority accuracy | **0.90** (0.96 on 28 unambiguous) | 0.70 (0.75) | ≥ 0.70 |
| Headline exact | 0.87 | **0.97** | |
| Achievements supported by CV text | — | **100 %** (120 facts) | |

**Iteration.** The first C2a run had years MAE **1.69**. Error analysis of the
five worst CVs found two generic bugs: (1) date ranges written with a fullwidth
hyphen "－" or a soft hyphen (present in **1 205 of 2 481** DB CVs) were not
parsed — one CV got 0 years instead of 16.2; (2) study periods in the Education
section were counted as work. After the fix MAE = 0.39 and the share of DB CVs
with 0 years fell from 2.5 % to 1.3 %. Because the fix was found on the same 30
CVs, the DB-wide change is the evidence that it generalises. Remaining largest
error: "Honor Roll 1996-2009" (an award period) is read as a job.

**Interpretation.** The LLM is *not* better at everything: it is precise but
lists far fewer skills (recall 0.37) and is worse at date arithmetic; rules are
better for structured, countable facts. The LLM is better at headlines and is the
only source of grounded achievements. **Design decision:** ranking uses C2a for
all CVs; C2b is used only for shortlisted CVs to supply roles and achievements
to the outreach writer. Seniority: 2 of 30 headlines contain keywords of two
levels ("Assistant Manager"); the guideline did not define precedence — reported
separately instead of changing the gold labels.

**Limitations.** Silver labels come from one LLM annotator, not humans; skill
F1 is computed only over the shared vocabulary (C2b's free-form skills outside it
are ignored); n = 30.

## C3 Hybrid retriever
**Script:** `eval/eval_retrieval.py` · **Data:** fit benchmark (`eval/gold/README.md`)
· **Encoder:** `BAAI/bge-small-en-v1.5`, 200-word chunks, max-pool · **Fusion:** RRF (k=60).

**Why these metrics.** The retriever's job is *recall*: it must put good
candidates into the top-50 handed to the ranker, so Recall@50 is the main metric;
MRR shows how early the first good candidate appears. Pool NDCG@10 (graded
0/1/2) measures ordering quality on fully labelled pools and is the baseline the
C4 ranker must beat.

### Iteration 1 — query = raw JD text
| Method | Recall@10 | Recall@20 | Recall@50 | MRR | Pool NDCG@10 |
|---|---|---|---|---|---|
| Random | 0.021 | 0.042 | 0.105 | — | 0.574 |
| BM25 | 0.044 | 0.087 | 0.231 | 0.194 | 0.670 |
| Dense | 0.052 | 0.145 | 0.361 | 0.198 | 0.709 |
| Hybrid | 0.049 | 0.117 | 0.316 | 0.236 | 0.700 |

Error analysis:
- **Labels are sparse.** On average 43 of the top-50 retrieved resumes have *no
  label* for that JD (each resume is labelled for ~3.7 of 71 JDs) and are counted
  as misses, so corpus Recall@k is a lower bound. Example: "Senior Tax
  Accountant" JD — Recall@50 = 0, yet pool NDCG@10 = 1.0: labelled candidates
  are ordered perfectly, but unlabelled accountants fill the top-50.
- **Query noise.** The full JD contains company marketing and benefits text
  ("X is Alphabet's moonshot factory…") that dilutes the query.

### Iteration 2 — query built from LLM-extracted JobSpec
Each JD → `JobSpec` with `prompts/jobspec_extractor.md` v1 (`gemini-3.5-flash-lite`,
71 calls); query = title + seniority + must-have skills (×2) + nice-to-have + domain.

| Method | Recall@10 | Recall@20 | Recall@50 | MRR | Pool NDCG@10 |
|---|---|---|---|---|---|
| BM25 | 0.098 | 0.203 | 0.397 | 0.268 | 0.728 |
| Dense | **0.145** | **0.248** | 0.445 | 0.283 | **0.757** |
| **Hybrid (chosen)** | 0.116 | 0.226 | **0.455** | **0.322** | 0.748 |

n = 476 resumes; Recall/MRR over 28 JDs; NDCG over 61 JDs.

Paired bootstrap over JDs (5 000 resamples), hybrid JobSpec vs hybrid raw JD:
- Recall@50 **+0.139**, 95 % CI [+0.052, +0.248]; 14 wins / 11 ties / 3 losses.
- Pool NDCG@10 **+0.049**, 95 % CI [+0.009, +0.089]; 35 / 11 / 15.
- vs BM25 on raw JD (the classic keyword baseline): Recall@50 +0.224, CI
  [+0.119, +0.345]; NDCG@10 +0.079, CI [+0.033, +0.128].
- Reproduced by `eval/eval_retrieval.py` → `significance` in `eval/results/retrieval.json`.

**Conclusions.**
- Structured queries are the single biggest win: removing boilerplate raises
  Recall@50 by 44 % relative; the improvement is statistically significant.
- Hybrid is chosen as default: best Recall@50 (what the ranker needs) and best
  MRR. Dense alone is slightly better at the very top (Recall@10, NDCG), which is
  the ranker's job anyway.
- Recall@50 = 0.455 is still below the 0.80 target, but the target assumed dense
  labels; with sparse labels this number is a lower bound. Remaining fix to
  estimate true precision: LLM-judge a sample of unlabelled top-10 hits.

## C4 Scorecard ranker
**Script:** `eval/eval_ranker.py` → `eval/results/ranker.json` · **Data:** labelled
pools of 61 benchmark JDs (1 703 candidate–JD pairs), graded labels 0/1/2.

**Why these metrics.** The ranker decides who the recruiter sees first, so
NDCG@10 (graded, position-aware) is the main metric; P@5 (share of "Good Fit" in
the first five cards) is what the recruiter experiences; Spearman between the
score and the label checks that the *whole* score is calibrated, not just the top.
The knock-out rule is evaluated separately as a classifier (rejection rate per class).

| Method | NDCG@10 | P@5 | Spearman |
|---|---|---|---|
| Random | 0.575 | 0.223 | 0.00 |
| C3 hybrid retrieval (baseline) | 0.748 | 0.266 | 0.34 |
| Dense retrieval only | 0.755 | 0.266 | 0.36 |
| Scorecard v1 (original business weights) | 0.747 | 0.285 | 0.34 |
| Scorecard v1 + knock-out (< 50 % must-haves) | 0.745 | 0.275 | 0.34 |
| **Scorecard v2 (revised weights, default)** | **0.808** | **0.295** | **0.40** |
| Weights auto-tuned, 2-fold CV (held-out) | 0.800 | — | — |

Scorecard v2 vs C3 baseline: NDCG@10 **+0.060**, 95 % CI [+0.012, +0.107]
(paired bootstrap over JDs). v1 vs C3: −0.001, CI [−0.054, +0.050] — not significant.
_Numbers after the C2a date-parsing fix (see C2 section); before the fix v2 was 0.814 and CV 0.786._

**Honest estimate.** v2 weights were chosen *after* seeing v1 results on the same
61 JDs, so 0.808 is optimistic. The unbiased estimate is the 2-fold
cross-validation: weights tuned on one half of the JDs and tested on the other
give **0.800** — still clearly above the 0.748 baseline. The tuned weights are
unstable between folds (experience 0.4–0.6, semantic 0.3–0.4), a sign that
61 JDs are few; the hand-set v2 is a business-readable compromise between them.

**Ablation (v2, drop one criterion, weights renormalised):**

| Without | NDCG@10 | Δ | Interpretation |
|---|---|---|---|
| experience | 0.752 | −0.056 | strongest signal: labels track years of relevant work |
| semantic | 0.778 | −0.030 | catches skills described in other words |
| title | 0.800 | −0.008 | |
| domain | 0.798 | −0.010 | |
| seniority | 0.806 | −0.002 | regex-derived level is coarse |
| must_have | 0.805 | −0.003 | small NDCG effect, but P@5 drops 0.295 → 0.272; kept for P@5 and explainability |
| nice_to_have | 0.814 | +0.006 | slightly noisy; candidate for removal |

**Knock-out rule — failed, disabled by default.** At the 50 % threshold it
rejects 87 % of Good Fit, 90 % of Potential and 90 % of No Fit candidates. Even
at the most lenient threshold (any must-have found) No Fit is rejected only 1.28×
more often than Good Fit. Root cause: in Good Fit CVs only 19 % of the JD's
must-have skills are written literally (e.g. "kubernetes" appears in 4 of 66
Good Fit CVs for JDs requiring it). This is not a text-processing bug — substring
search finds the same — but a property of the labels: annotators judged overall
role fit, and real CVs omit many listed tools. Business consequence: a strict
must-have filter would hide most good candidates, so it became an opt-in strict
mode, and missing must-haves are shown as a warning instead.

## C4b LLM re-ranker
**Script:** `eval/eval_reranker_llm.py` → `eval/results/reranker_llm.json` ·
`gemini-3.5-flash-lite`, prompt `llm_reranker` v1, one call per JD rating the
top-10 of scorecard v2 (shuffled with a fixed seed against position bias);
61 JDs, 567 rated candidates (pools smaller than 10 are rated fully).

| Ordering of the top-10 | NDCG@10 | P@5 |
|---|---|---|
| **Scorecard v2** | **0.808** | **0.295** |
| LLM `overall` | 0.801 | 0.289 |
| Blend 50/50 (scorecard position + LLM) | 0.806 | 0.292 |

LLM vs scorecard: −0.007, 95 % CI [−0.025, +0.008] — **no improvement**.
_Ratings were collected on the top-10 before the C2a fix; after it 98 % of the
new top-10 members are rated (unrated ones sort last in the LLM order)._

| Quality of the ratings themselves | Value |
|---|---|
| Mean `overall` by label (No / Potential / Good) | 1.85 / 2.52 / 2.85 — monotone |
| Spearman(`overall`, label) | 0.37 |
| Evidence quote verbatim in CV | 93 % |
| Candidates rated (coverage of current top-10) | 98 % |

**Interpretation.** The LLM's judgement is sensible (monotone in the label,
grounded quotes) but adds no ranking signal beyond the scorecard: within an
already well-ordered top-10 there is little headroom, and the model sees only a
450-word excerpt of each CV. **Decision:** keep the scorecard order
(`RERANK_REORDER = False`) and use the LLM ratings as the explanation layer on
candidate cards; quotes that are not found verbatim in the CV (7 %) are dropped
before display. Possible next fix: give the LLM the full CV or the evidence
snippets from the scorecard instead of the CV head.
