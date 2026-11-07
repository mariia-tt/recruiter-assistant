---
name: component-eval
description: Design and run the evaluation of one pipeline component — choose metrics, build a gold set, compare against a baseline, analyse errors. Use whenever a component is created or changed.
---

# Evaluating a component

1. **Choose metrics that match the component's job** (justify the choice in the
   report):
   | Component type | Metrics |
   |---|---|
   | Slot filling / dialog | slot precision/recall/F1, completion rate, turns to completion |
   | Information extraction | field-level P/R/F1 (normalised), MAE for numbers, accuracy for categories |
   | Retrieval | Recall@k, MRR |
   | Ranking | NDCG@k (graded labels), Precision@k, Spearman vs labels |
   | Generation | faithfulness (supported claims / claims), hallucination rate, constraint compliance, LLM-judge score |
   | LLM-as-judge itself | agreement with human labels (accuracy, Cohen's kappa) |
2. **Gold set** in `eval/gold/<component>.*` plus a section in
   `eval/gold/README.md`: source, size, how labelled, by whom. Prefer labels
   from open datasets; hand-label small sets (20–50 items) when needed.
3. **Baseline.** Always compare with at least one simple baseline (e.g. BM25
   only, keyword matching, random) so numbers mean something.
4. **Script** `eval/eval_<component>.py`: deterministic (temperature 0, seed,
   cache), writes `eval/results/<component>.json` with metrics, config, prompt
   version, model name, n, timestamp.
5. **Error analysis.** Look at the 5 worst cases. In `docs/05_evaluation.md`
   write: metric table, what high vs low scores look like, *why* errors happen,
   and the concrete next fix.
6. If a metric is below the success criterion in
   `docs/01_business_understanding.md` → loop back (log it with `crisp-dm`).
