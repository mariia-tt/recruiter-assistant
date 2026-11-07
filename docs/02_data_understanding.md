# Phase 2 — Data Understanding

Produced with `scripts/download_data.py` and `scripts/explore_data.py`.

## Datasets

| | Candidate DB | Benchmark |
|---|---|---|
| Source | `opensporks/resumes` (HF mirror of Kaggle *Resume Dataset*, livecareer.com CVs) | `cnamuangtoun/resume-job-description-fit` (test split) |
| Role in project | searchable candidate database in the app | retrieval & ranking evaluation (has human fit labels) |
| Size | 2 484 CVs, 24 categories | 1 759 resume–JD pairs, 477 unique resumes, 71 JDs |
| Columns | `ID, Resume_str, Resume_html, Category` | `resume_text, job_description_text, label` |
| Labels | job category (balanced ~100–120, except BPO 22, AUTOMOBILE 36, AGRICULTURE 63) | No Fit 857 / Potential Fit 444 / Good Fit 458 |
| CV length (words p0/p50/p95/max) | 0 / 757 / 1 399 / 5 190 | 84 / 618 / 1 449 / 2 738 |

Why not generate synthetic CVs: both open datasets contain real CV text, and the
benchmark gives independent human relevance labels — this is stronger evidence
for the ranking metrics than labels we would create ourselves.

## Quality findings → preparation rules

| Finding | Evidence | Rule for Phase 3 |
|---|---|---|
| CVs are anonymised: employer = "Company Name", no candidate names | 2 483 / 2 484 CVs contain the placeholder | Outreach messages must not use a name or employer; greet neutrally, refer to roles/skills/achievements instead |
| Residual PII | 19 CVs with e-mails, 48 with phone numbers | Mask e-mails/phones before indexing and before sending text to the LLM |
| Empty / tiny CVs | 1 empty, min 21 chars | Drop CVs with < 50 words |
| Exact duplicates | 2 in candidate DB; benchmark reuses 477 resumes across JDs | Dedupe the candidate DB by text hash; benchmark dedupe is not needed (pairs are by design) |
| Glued words from PDF extraction (`ExperienceData`, `inMay`) | common in benchmark; in candidate DB mostly false positives (`PowerPoint`) | Insert space at lower→Upper boundaries only before known section words (Experience, Education, Skills, Summary, Highlights…) and month names |
| Very long CVs | p95 ≈ 1 400 words, max 5 190 | Truncate to ~2 000 words for LLM profiling (keeps > 95 % of CVs intact) |
| Benchmark: only 28 of 71 JDs have ≥ 1 "Good Fit" | label table | Ranking NDCG computed on graded labels (Good=2, Potential=1, No=0) over JDs with ≥ 1 non-zero label; Recall@k on the 28 JDs with Good Fit |
| Categories are broad (e.g. INFORMATION-TECHNOLOGY mixes helpdesk and developers) | category list | Do not filter by category; use it only as a weak domain signal |
