# Annotation guideline — are outreach-message claims supported by the CV? (validates C6)

Input: `eval/gold/claims_sample.jsonl` (fields `item`, `cv_id`, `claim`) and
`eval/gold/claims_cvs.jsonl` (fields `cv_id`, `text`).
Output: `eval/gold/claims_gold.jsonl`, one object per claim, same order:
`{"item": 0, "supported": true|false, "evidence": "verbatim CV fragment or null", "notes": ""}`

- `supported` = true if the CV states the claim about the candidate. Paraphrase is fine;
  numbers, years, tool names and job titles must match what the CV says.
- A claim that overstates the CV (e.g. "used Angular extensively" when the CV says
  "basic understanding of AngularJS"; "led a team" when the CV says "team member") is
  false. A total number of years that the CV never states is false unless it can be
  read directly from a single explicit statement such as "20+ years of experience".
- Opinions attached to a fact ("makes you a strong fit") do not matter — judge the fact.
- `evidence`: the shortest verbatim fragment (≤ 20 words) supporting the claim; null if false.
- Judge each claim independently from the CV text only.
