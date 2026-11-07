# Phase 5 — Evaluation

One section per component. Raw numbers live in `eval/results/*.json`.

## C0 Data cleaning
See `eval/results/data_cleaning.json` and the Phase 3 entry in
`docs/crisp_dm_log.md`: residual PII 79 → 2 documents (both manually verified
false positives — licence numbers), duplicates 2 → 0, glued words removed
(verified by unit tests).

## C3 Hybrid retriever
**Script:** `eval/eval_retrieval.py` · **Data:** fit benchmark (`eval/gold/README.md`)
· **Query:** raw JD text · **Encoder:** `BAAI/bge-small-en-v1.5`, 200-word chunks, max-pool.

**Why these metrics.** The retriever's job is *recall*: it must put good
candidates into the top-50 handed to the ranker, so Recall@k is the main metric;
MRR shows how early the first good candidate appears. Pool NDCG@10 (graded
0/1/2) measures ordering quality on fully labelled pools and is the baseline the
C4 ranker must beat.

| Method | Recall@10 | Recall@20 | Recall@50 | MRR | Pool NDCG@10 |
|---|---|---|---|---|---|
| Random | 0.021 | 0.042 | 0.105 | — | 0.574 |
| BM25 | 0.044 | 0.087 | 0.231 | 0.194 | 0.670 |
| Dense (bge-small) | **0.052** | **0.145** | **0.361** | 0.198 | **0.709** |
| Hybrid (RRF) | 0.049 | 0.117 | 0.316 | **0.236** | 0.700 |

n = 477 resumes; Recall/MRR over 28 JDs; NDCG over 69 JDs.

**Reading the numbers.**
- All methods beat random by 2–3×; dense beats BM25 on every metric —
  vocabulary mismatch (synonyms, paraphrased duties) matters for CVs.
- Hybrid gives the best MRR (first good candidate appears earlier) but slightly
  lower recall than dense alone: BM25 pulls in CVs that share boilerplate
  words with the JD.
- **Recall@50 = 0.32 is far below the 0.80 target, but it is a lower bound.**
  Error analysis: on average 43 of the top-50 retrieved resumes have *no label*
  for that JD (labels cover ~3.7 JDs per resume) and are counted as misses.
  Example: "Senior Tax Accountant" JD — Recall@50 = 0, yet pool NDCG@10 = 1.0:
  the retriever orders the labelled candidates perfectly, but other
  (unlabelled) accountants from the corpus fill the top-50.
- Second cause: the query is the full JD including company marketing and
  benefits text ("X is Alphabet's moonshot factory…"), which dilutes it.

**Next fixes (loop-back to Modeling):**
1. Query from a structured `JobSpec` (title + must-have skills) extracted by the
   LLM instead of raw JD text — removes boilerplate.
2. Estimate true precision of unlabelled hits with an LLM judge on a small
   sample, so the recall lower bound can be complemented by a realistic number.
