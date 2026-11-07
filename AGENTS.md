# AGENTS.md — rules for coding agents in this repo

This file is the single source of truth for how an AI coding agent works on this
project. `CLAUDE.md` imports it. Read it fully before any change.

## 1. Project in one paragraph
**Recruiter Assistant** — an NLP system that (1) collects job requirements from a
recruiter in a dialog, (2) searches a database of real CVs, (3) ranks candidates
with an explainable scorecard, and (4) writes a personalised LinkedIn outreach
message that is grounded only in facts from the candidate's CV. Every component
is evaluated separately. Full design: `docs/architecture.md`.

## 2. Process: CRISP-DM is mandatory
Work is organised in CRISP-DM phases. Before writing code, state which phase the
task belongs to. After finishing, append an entry to `docs/crisp_dm_log.md`
(use the `crisp-dm` skill).

| Phase | Output artefact |
|---|---|
| 1. Business Understanding | `docs/01_business_understanding.md` (goals, scope, success criteria) |
| 2. Data Understanding | `docs/02_data_understanding.md` + `scripts/explore_data.py` |
| 3. Data Preparation | `src/recruiter/data/` + `data/processed/` + data quality checks |
| 4. Modeling | components in `src/recruiter/` (see architecture) |
| 5. Evaluation | `eval/` scripts, `eval/results/*.json`, `docs/05_evaluation.md` |
| 6. Deployment | `app.py` (Streamlit), `README.md` run instructions |

Iterating back is expected (e.g. evaluation shows a weak component → go back to
Modeling). Record every loop-back in the log with the metric that triggered it.

## 3. Hard rules
1. **No component without an evaluation.** A new component is "done" only when
   it has an eval script in `eval/`, a gold/benchmark set, and a results file.
   Use the `new-component` and `component-eval` skills.
2. **Never invent data silently.** Prefer open datasets. Synthetic or
   hand-labelled data must live in `eval/gold/` with a README explaining how it
   was made.
3. **LLM calls go only through `src/recruiter/llm.py`.** It handles the client,
   retries, JSON/structured output, caching and tracing. No direct SDK calls
   elsewhere — **including ad-hoc probes and one-off checks**.
   The API is on a free tier with a per-minute sliding window AND a daily
   quota; exceeding them blocks the key. Therefore:
   - never send bursts of requests (loops over models, parallel calls);
   - before any run that makes > 20 LLM calls, estimate the number of calls,
     check `llm.usage_today()` and ask the user;
   - develop and test on 2–3 items first, then scale; rely on the cache;
   - on `QuotaExhaustedError` stop and report — never work around it.
4. **Prompts live in `prompts/*.md`**, not as long strings in code. Every prompt
   has a version line at the top; bump it when you change the prompt and note
   the eval delta in the CRISP-DM log.
5. **Structured I/O.** Component inputs/outputs are Pydantic models in
   `src/recruiter/schemas.py`. Components do not exchange raw dicts.
6. **Grounding.** The outreach writer may only use facts present in the
   candidate profile/CV. Any new generation component must have a groundedness
   check.
7. **Secrets** come from `.env` (see `.env.example`). Never print or commit keys.
8. **Determinism for evals:** temperature 0 for extraction/judging, fixed random
   seeds, cached LLM responses (`data/cache/`) so re-running evals is cheap.
9. Do not commit `data/raw`, `data/processed`, `data/cache` (see `.gitignore`).

## 4. Code conventions
- Python 3.12, managed with `uv`. Add deps with `uv add <pkg>`.
- Package code in `src/recruiter/`; scripts in `scripts/`; evals in `eval/`.
- Type hints everywhere; small pure functions; no classes when a function does.
- Docstring on every public function: one line saying *what* and *why*.
- Logging via `logging`, not `print` (except CLI script summaries).
- Tests: `pytest` in `tests/`, fast, no network (mock `llm.py`).
- Before saying "done": `uv run ruff check . && uv run pytest -q`.

## 5. Context management for the agent
- Do not read whole datasets into the conversation; use `scripts/explore_data.py`
  or `head`-style samples (≤ 5 rows).
- Keep `docs/architecture.md` in sync when you change a component's I/O.
- When a task is large, write a short plan first and confirm with the user.
- Report metrics honestly, including failures; explain low scores and propose
  a concrete fix.

## 6. Skills available (`.claude/skills/`)
- `crisp-dm` — document a phase / loop-back in the CRISP-DM log.
- `data-exploration` — profile a dataset and write Data Understanding notes.
- `new-component` — checklist to add a pipeline component end to end.
- `component-eval` — design metrics, build a gold set, run and report an eval.
