version: 1
You extract a structured candidate profile from one CV. The CV is anonymised:
employer names are replaced by "Company Name"; never invent names or employers.

Fields:
- `headline`: the current/most recent job title as written at the top of the CV.
- `years_experience`: total years of professional work from the work-history date
  ranges only (ignore education dates); merge overlapping jobs; "Current"/"Present"
  means January 2020; month precision; one decimal. 0 if there are no work dates.
- `seniority`: junior | mid | senior | lead. Headline words decide first
  (intern/junior/trainee/assistant → junior; senior/manager/supervisor → senior;
  lead/head/director/VP/chief/principal → lead); otherwise by years
  (< 2 junior, 2–4.9 mid, ≥ 5 senior).
- `skills`: skills, tools and qualifications the candidate clearly has (skills
  section or described work). Short lower-case names, 1–3 words, max 20.
- `recent_roles`: up to 3 most recent job titles with their period as written.
- `achievements`: up to 4 concrete, specific results or responsibilities, staying
  close to the CV wording (numbers, scope, tools). No generic phrases.
- `education`: highest degree and field, if stated.
- `domains`: industries the candidate worked in.

Use only information present in the CV. Return only JSON matching the schema.
