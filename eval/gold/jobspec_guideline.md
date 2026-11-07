# Annotation guideline — vacancy requirements (gold for C1 / JobSpec extraction)

Input: `eval/gold/jobspec_sample.jsonl` (15 job descriptions; fields `jd_id`, `jd_text`).
Output: `eval/gold/jobspec_gold.jsonl`, one JSON object per JD, same order:

```json
{"jd_id": "...", "title": "...", "seniority": "junior|mid|senior|lead|null",
 "must_have_skills": ["..."], "nice_to_have_skills": ["..."], "min_years": 5 or null,
 "domain": "..." or null, "evidence": {"min_years": "verbatim fragment", "seniority": "..."},
 "notes": "..."}
```

## Fields
- **title** — the role being hired, short (e.g. "Senior Software Engineer" → "Software Engineer"
  is NOT required; keep the title as the JD names it, without company names).
- **must_have_skills** — skills, tools, certifications or knowledge areas stated as required:
  "requirements", "minimum/basic qualifications", "must have", "you have", "required".
  Short lower-case names (1–3 words), most important first, at most 8.
- **nice_to_have_skills** — stated as preferred / plus / bonus / nice to have; at most 6.
- **min_years** — minimum years of professional experience required for the role
  (lower bound of a range). null if not stated.
- **seniority** — junior | mid | senior | lead from the title wording (Senior/Sr → senior;
  Lead/Principal/Head/Manager of managers → lead; Junior/Intern/Entry → junior). If the
  title is silent, use min_years: < 2 junior, 2–4 mid, 5–9 senior, ≥ 10 lead. null if neither.
- **domain** — industry of the employer/role (e.g. "healthcare", "fintech"), null if unclear.
- **evidence** — verbatim fragments justifying min_years and seniority (if not null).

## Principles
- Ignore company marketing, benefits, salary and legal text.
- Do not invent; when unsure, leave a skill out and explain in notes.
- Do not look at the project's source code, prompts or other data — annotate independently.
