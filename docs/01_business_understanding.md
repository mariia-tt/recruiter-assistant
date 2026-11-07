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
