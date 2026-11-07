---
name: new-component
description: Checklist for adding or changing a pipeline component (requirements collector, CV profiler, retriever, ranker, outreach writer, grounding checker, or a new one). Use before writing the component's code.
---

# Adding a pipeline component

Do these steps in order; do not skip the eval.

1. **Contract first.** Define input/output Pydantic models in
   `src/recruiter/schemas.py`. Write a 3-line spec in `docs/architecture.md`:
   purpose, input, output, model/tool calls.
2. **Prompt** (if the component calls an LLM): create `prompts/<component>.md`
   with `version: 1` on the first line. Ask for JSON matching the schema.
3. **Code** in `src/recruiter/<component>.py`:
   - one public entry function, typed in/out with the schemas;
   - LLM access only via `recruiter.llm` (structured output + cache + trace);
   - no hidden global state; config (weights, k, model) via `recruiter.config`.
4. **Unit test** in `tests/test_<component>.py` with mocked LLM.
5. **Evaluation**: run the `component-eval` skill → `eval/eval_<component>.py`
   and `eval/results/<component>.json`.
6. **Wire it** into `src/recruiter/pipeline.py` and the Streamlit app.
7. Update the data-flow diagram in `docs/architecture.md` if I/O changed.
8. `uv run ruff check . && uv run pytest -q`, then log with `crisp-dm`.
