"""Recruiter Assistant — Streamlit UI (Phase 6: Deployment).

Run: uv run streamlit run app.py
"""

import json

import streamlit as st

from recruiter import config, llm
from recruiter.collector import next_turn
from recruiter.pipeline import explain_top, load_db, search_candidates
from recruiter.schemas import JobSpec

st.set_page_config(page_title="Recruiter Assistant", page_icon="🧭", layout="wide")

GREETING = (
    "Hi! Describe the vacancy you are hiring for — role, must-have skills, experience, level. "
    "You can also paste a full job description."
)
CRITERIA_LABELS = {
    "must_have": "Must-have skills", "experience": "Experience", "title": "Role / title",
    "nice_to_have": "Nice-to-have", "seniority": "Seniority", "domain": "Domain",
    "semantic": "Overall semantic match",
}


@st.cache_resource(show_spinner="Loading candidate database and search index…")
def db():
    return load_db()


def init_state() -> None:
    defaults = {"messages": [{"role": "assistant", "content": GREETING}], "spec": None,
                "results": None, "ratings": {}}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def sidebar() -> float | None:
    """Settings and LLM budget; returns the knock-out threshold (None = off)."""
    st.sidebar.header("Settings")
    strict = st.sidebar.toggle(
        "Strict mode: hide candidates with < 50 % of must-haves", value=False,
        help="Off by default: on labelled data this rule rejected most good candidates.",
    )
    used = llm.usage_today(config.LLM_MODEL)
    st.sidebar.metric("LLM calls today", f"{used} / {config.LLM_RPD}")
    st.sidebar.caption(f"Model: `{config.LLM_MODEL}` · CVs in DB: {len(db().texts)}")
    if st.sidebar.button("New search"):
        for k in ("messages", "spec", "results", "ratings"):
            st.session_state.pop(k, None)
        st.rerun()
    return 0.5 if strict else None


def chat(knockout: float | None) -> None:
    """C1 dialog: collect requirements, then run the search pipeline."""
    for m in st.session_state.messages:
        st.chat_message(m["role"]).markdown(m["content"])
    prompt = st.chat_input("Describe the vacancy…")
    if not prompt:
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)
    with st.spinner("Understanding requirements…"):
        try:
            turn = next_turn(st.session_state.messages)
        except llm.QuotaExhaustedError as e:
            st.error(f"Daily LLM budget used up: {e}")
            return
    st.session_state.messages.append({"role": "assistant", "content": turn.reply})
    st.session_state.spec = turn.spec
    if turn.done:
        with st.spinner("Searching and ranking candidates…"):
            st.session_state.results = search_candidates(turn.spec, knockout=knockout)
            st.session_state.ratings = {}
    st.rerun()


def criteria_table(c) -> None:
    for x in c.criteria:
        if x.weight == 0:
            continue
        cols = st.columns([2, 3, 1])
        cols[0].caption(f"{CRITERIA_LABELS.get(x.name, x.name)} · w={x.weight:.2f}")
        cols[1].progress(min(max(x.score, 0.0), 1.0))
        cols[2].caption(f"{x.score:.2f}")


def candidate_card(rank: int, c, spec: JobSpec) -> None:
    d = db()
    p = d.profiles[c.id]
    flag = " ⚠️ near miss" if c.knocked_out else ""
    title = (f"#{rank} · {p.headline or 'Untitled'} · score {c.total:.2f} · "
             f"{p.years_experience:g} y · {p.seniority}{flag}")
    with st.expander(title, expanded=rank <= 3):
        left, right = st.columns([3, 2])
        with left:
            criteria_table(c)
        with right:
            st.caption(f"CV id `{c.id}` · category {d.categories[c.id]}")
            if c.matched_must:
                st.success("Has: " + ", ".join(c.matched_must))
            if c.missing_must:
                st.warning("Not found in CV: " + ", ".join(c.missing_must))
        must = next((x for x in c.criteria if x.name == "must_have"), None)
        if must and must.evidence:
            st.markdown("**Evidence from CV**")
            for e in must.evidence[:4]:
                st.caption(e)
        r = st.session_state.ratings.get(c.id)
        if r:
            st.markdown(
                f"**LLM assessment** — overall {r.overall}/5 · skills {r.skills_fit}/5 · "
                f"experience {r.experience_fit}/5 · role {r.role_fit}/5"
            )
            if r.evidence_quote:
                st.markdown(f"> {r.evidence_quote}")
            if r.concern:
                st.caption(f"Concern: {r.concern}")
        with st.popover("Show CV text"):
            st.text(d.texts[c.id][:4000])


def results_panel() -> None:
    spec: JobSpec | None = st.session_state.spec
    if spec is None:
        st.info("Requirements will appear here as you chat.")
        return
    st.subheader("Vacancy (JobSpec)")
    missing = spec.missing_mandatory()
    st.json(json.loads(spec.model_dump_json(exclude_none=True)), expanded=False)
    if missing:
        st.caption("Still missing: " + ", ".join(missing))
    results = st.session_state.results
    if not results:
        return
    shortlist = results[:10]
    st.subheader(f"Top candidates ({len(results)} retrieved and ranked)")
    if not st.session_state.ratings and st.button(
        "Explain top-10 with LLM (1 call)", help="Recruiter-style ratings with verified quotes"
    ):
        with st.spinner("Asking the LLM for a structured assessment…"):
            try:
                _, st.session_state.ratings = explain_top(spec, results)
            except llm.QuotaExhaustedError as e:
                st.error(str(e))
        st.rerun()
    for i, c in enumerate(shortlist, 1):
        candidate_card(i, c, spec)


def main() -> None:
    init_state()
    knockout = sidebar()
    st.title("🧭 Recruiter Assistant")
    chat_col, res_col = st.columns([2, 3])
    with chat_col:
        chat(knockout)
    with res_col:
        results_panel()


main()
