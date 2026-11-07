# Annotation guideline — CV profiles (silver standard for C2a / C2b)

Input: `eval/gold/cv_profiles_sample.jsonl` (30 cleaned CVs; fields `id`, `category`, `text`).
Output: `eval/gold/cv_profiles.jsonl`, one JSON object per CV, same order:

```json
{"id": "...", "headline": "...", "years_experience": 0.0, "seniority": "junior|mid|senior|lead",
 "skills": ["..."], "skill_evidence": {"skill": "verbatim fragment from the CV"}, "notes": "..."}
```

## Fields
- **headline** — the candidate's current/most recent job title as written at the
  top of the CV (usually the first words before "Summary", "Profile",
  "Experience", etc.). Copy it, do not rephrase.
- **years_experience** — total years of *professional work* (jobs, internships,
  freelancing). Use the date ranges in the work-history part only; ignore
  education dates and single years without a range. Overlapping jobs count once
  (merge overlapping periods). "Current" / "Present" = **2020-01**. Month-level
  precision: 01/2015 to 07/2016 = 1.5 years. If no work dates exist, 0. Round to
  one decimal.
- **seniority** — from the headline wording first: intern / junior / jr /
  trainee / assistant / entry-level → `junior`; senior / sr / manager /
  supervisor → `senior`; lead / head / director / VP / chief / principal /
  partner / owner → `lead`. If the headline has no such word, use the years:
  < 2 → `junior`, 2–4.9 → `mid`, ≥ 5 → `senior`.
- **skills** — every skill from `eval/gold/skill_vocabulary.txt` (use those exact
  names) that the CV **clearly shows the candidate has**: listed in a skills
  section, or described in work they did. Synonyms and specific tools count for
  the general skill (e.g. "pandas" → `python`, "MySQL" → `sql`, "QuickBooks"
  bookkeeping → `accounting` and `quickbooks`, "supervised 12 staff" →
  `team leadership`). Do NOT add a skill only because the job category implies it.
- **skill_evidence** — for each skill, the shortest verbatim fragment (≤ 15 words)
  from the CV that justifies it.
- **notes** — anything ambiguous (e.g. "dates unclear, estimated").

## Principles
- Judge only from the CV text; never guess.
- Be consistent across CVs; when in doubt, leave a skill out and mention it in notes.
- Do not look at the project's source code or prompts — annotate independently.
