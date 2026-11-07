# Phase 1 — Business Understanding

## Problem
A recruiter spends most of the time on two manual steps: screening many CVs
against a vacancy and writing personalised first-contact messages. Keyword
filters in ATS tools miss good candidates (synonyms, different wording) and
generic outreach messages get low reply rates.

## Goal of the system
Given a conversation with a recruiter, the assistant:
1. **Collects requirements** for a vacancy (asks follow-up questions until the
   mandatory fields are known).
2. **Finds and ranks** the best candidates from a CV database using transparent
   business rules (a weighted scorecard inspired by structured interview
   evaluation forms), and explains each score with evidence from the CV.
3. **Writes a LinkedIn outreach message** for each selected candidate that is
   personalised and grounded only in facts from that candidate's CV.

## In scope
- English-language CVs and vacancies.
- Requirements: job title, must-have skills, nice-to-have skills, minimum years
  of experience, seniority, domain/industry, location / remote (optional).
- Scorecard criteria (weights configurable):
  | Criterion | Default weight | How it is measured |
  |---|---|---|
  | Must-have skills coverage | 0.40 | share of must-have skills found in profile (with synonyms) |
  | Relevant experience | 0.20 | years of experience vs required minimum |
  | Role / title relevance | 0.15 | semantic similarity of recent titles to vacancy title |
  | Nice-to-have skills | 0.10 | share of nice-to-have skills found |
  | Seniority match | 0.10 | exact / adjacent / mismatch |
  | Domain match | 0.05 | industry overlap |
  - **Knock-out rule:** candidates covering < 50 % of must-have skills are not
    shortlisted (shown separately as "near misses").

  > **Revision after evaluation (2026-11-07, v2).** On the labelled benchmark the
  > knock-out rule rejected 87 % of "Good Fit" candidates — almost as many as
  > "No Fit" (90 %): CVs of good candidates often do not name every listed tool.
  > A hard rule that drops good people is worse for the business than a soft
  > signal, so the knock-out is **off by default** (recruiter can enable a strict
  > mode); missing must-haves are shown as a warning on each candidate card.
  > Weights v2 (used by default):
  >
  > | Criterion | v1 | v2 | Reason for change |
  > |---|---|---|---|
  > | Must-have skills | 0.40 | 0.25 | literal matching misses skills described in other words |
  > | Relevant experience | 0.20 | 0.25 | strongest single predictor in ablation |
  > | Role / title relevance | 0.15 | 0.10 | |
  > | Nice-to-have skills | 0.10 | 0.05 | no measurable effect |
  > | Seniority match | 0.10 | 0.05 | rule-derived level is a coarse signal |
  > | Domain match | 0.05 | 0.10 | helps in ablation |
  > | **Semantic match** (new) | — | 0.20 | overall CV ↔ vacancy similarity; catches paraphrased skills |
  >
  > Evidence and the unbiased (cross-validated) estimate: `docs/05_evaluation.md`.
- Human in the loop: the recruiter reviews the shortlist and edits messages; the
  system never contacts candidates itself.

## Out of scope
- Real LinkedIn integration, sending messages, scheduling interviews.
- Salary prediction; any use of protected attributes (age, gender, ethnicity,
  photos). The scorecard ignores them by design — fairness requirement.

## Success criteria (measured in Phase 5)
| Component | Metric | Target |
|---|---|---|
| Requirements collector | slot F1 vs gold requirements | ≥ 0.80 |
| Requirements collector | dialogs completed in ≤ 4 assistant turns | ≥ 80 % |
| CV profiler (extraction) | skills F1 / years MAE / seniority accuracy | ≥ 0.75 / ≤ 1.5 y / ≥ 0.70 |
| Hybrid retriever | Recall@50 of "Good Fit" CVs | ≥ 0.80 |
| Scorecard ranker | NDCG@10 (graded fit labels) | better than BM25 baseline |
| Outreach writer | faithfulness (supported claims share) | ≥ 0.95 |
| Outreach writer | personalisation: ≥ 2 CV-specific facts, ≤ 120 words | ≥ 90 % of messages |
| LLM judge | agreement with human labels (Cohen's kappa) | ≥ 0.6 |
