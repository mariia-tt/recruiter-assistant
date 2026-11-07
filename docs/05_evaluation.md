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
