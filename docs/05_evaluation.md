# Phase 5 — Evaluation

One section per component. Raw numbers live in `eval/results/*.json`.

## C0 Data cleaning
See `eval/results/data_cleaning.json` and the Phase 3 entry in
`docs/crisp_dm_log.md`: residual PII 79 → 2 documents (both manually verified
false positives — licence numbers), duplicates 2 → 0, glued words removed
(verified by unit tests).

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
| Scorecard v1 (original business weights) | 0.754 | 0.285 | 0.36 |
| Scorecard v1 + knock-out (< 50 % must-haves) | 0.751 | 0.275 | 0.36 |
| **Scorecard v2 (revised weights, default)** | **0.814** | **0.292** | **0.43** |
| Weights auto-tuned, 2-fold CV (held-out) | 0.786 | — | — |

Scorecard v2 vs C3 baseline: NDCG@10 **+0.065**, 95 % CI [+0.017, +0.114]
(paired bootstrap over JDs). v1 vs C3: +0.006, CI [−0.048, +0.058] — not significant.

**Honest estimate.** v2 weights were chosen *after* seeing v1 results on the same
61 JDs, so 0.814 is optimistic. The unbiased estimate is the 2-fold
cross-validation: weights tuned on one half of the JDs and tested on the other
give **0.786** — still clearly above the 0.748 baseline. The tuned weights are
unstable between folds (experience 0.4–0.6, semantic 0.3 in both), a sign that
61 JDs are few; the hand-set v2 is a business-readable compromise between them.

**Ablation (v2, drop one criterion, weights renormalised):**

| Without | NDCG@10 | Δ | Interpretation |
|---|---|---|---|
| experience | 0.752 | −0.062 | strongest signal: labels track years of relevant work |
| semantic | 0.782 | −0.032 | catches skills described in other words |
| title | 0.803 | −0.011 | |
| domain | 0.806 | −0.008 | |
| seniority | 0.811 | −0.003 | regex-derived level is coarse |
| must_have | 0.814 | 0.000 | no NDCG effect, but P@5 drops 0.292 → 0.272; kept for P@5 and explainability |
| nice_to_have | 0.819 | +0.005 | slightly noisy; candidate for removal |

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
| **Scorecard v2** | **0.814** | **0.292** |
| LLM `overall` | 0.805 | 0.289 |
| Blend 50/50 (scorecard position + LLM) | 0.812 | 0.289 |

LLM vs scorecard: −0.009, 95 % CI [−0.027, +0.006] (12 wins / 21 ties / 28 losses) —
**no improvement**.

| Quality of the ratings themselves | Value |
|---|---|
| Mean `overall` by label (No / Potential / Good) | 1.85 / 2.52 / 2.85 — monotone |
| Spearman(`overall`, label) | 0.37 |
| Evidence quote verbatim in CV | 93 % |
| Candidates rated (coverage) | 100 % |

**Interpretation.** The LLM's judgement is sensible (monotone in the label,
grounded quotes) but adds no ranking signal beyond the scorecard: within an
already well-ordered top-10 there is little headroom, and the model sees only a
450-word excerpt of each CV. **Decision:** keep the scorecard order
(`RERANK_REORDER = False`) and use the LLM ratings as the explanation layer on
candidate cards; quotes that are not found verbatim in the CV (7 %) are dropped
before display. Possible next fix: give the LLM the full CV or the evidence
snippets from the scorecard instead of the CV head.
