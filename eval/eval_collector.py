"""Evaluate C1 requirements collector on simulated dialogs against an independent gold.

Gold: eval/gold/jobspec_gold.jsonl (15 JDs, independent LLM annotator, see eval/gold/README.md).
A simulator LLM plays the recruiter who holds the JD and reveals it only when asked.
Upper bound for comparison: JobSpec extracted from the full JD text (already cached).
Cost: <= 2 LLM calls per dialog turn, <= 4 turns per dialog. Usage: --limit 2
"""

import argparse
import json
import re
from datetime import UTC, datetime

import numpy as np

from recruiter import config, llm
from recruiter.collector import next_turn
from recruiter.jobspec import extract_jobspec
from recruiter.retrieval import embed
from recruiter.skills import canonicalize

SEMANTIC_THRESHOLD = 0.8  # bge-small cosine for "same skill, different wording"

GOLD = config.EVAL_DIR / "gold"
SIM = "recruiter_simulator"
OUT = config.EVAL_DIR / "results" / "collector_dialogs.jsonl"


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"\W+", canonicalize(s)) if t}


def soft_match(a: str, b: str) -> bool:
    """Same canonical skill, or token Jaccard >= 0.5 ("financial reporting" ~ "reporting")."""
    if canonicalize(a) == canonicalize(b):
        return True
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= 0.5


def semantic_match(a: str, b: str) -> bool:
    """Lexical match, or embedding cosine >= threshold ("restful apis" ~ "restful api design")."""
    if soft_match(a, b):
        return True
    ea, eb = embed([a, b])
    return float(ea @ eb) >= SEMANTIC_THRESHOLD


def skill_prf(pred: list[str], gold: list[str], match=soft_match) -> tuple[int, int, int]:
    """Greedy one-to-one matching -> (tp, fp, fn)."""
    left, tp = list(gold), 0
    for p in pred:
        hit = next((g for g in left if match(p, g)), None)
        if hit is not None:
            left.remove(hit)
            tp += 1
    return tp, len(pred) - tp, len(left)


def slot_scores(pred: dict, gold: dict) -> dict:
    py, gy = pred.get("min_years"), gold.get("min_years")
    title_ok = bool(pred.get("title")) and soft_match(pred["title"], gold["title"] or "")
    return {
        "title": title_ok,
        "seniority": pred.get("seniority") == gold.get("seniority"),
        "min_years": (py is None and gy is None) or (py is not None and gy is not None
                                                     and abs(py - gy) <= 0.5),
        "must": skill_prf(pred.get("must_have_skills", []), gold["must_have_skills"]),
        "nice": skill_prf(pred.get("nice_to_have_skills", []), gold["nice_to_have_skills"]),
        "must_sem": skill_prf(pred.get("must_have_skills", []), gold["must_have_skills"],
                              semantic_match),
        "nice_sem": skill_prf(pred.get("nice_to_have_skills", []), gold["nice_to_have_skills"],
                              semantic_match),
    }


def resolved(dialog: dict, gold: dict) -> bool:
    """Every mandatory scalar slot is filled, or empty because the gold says it is absent."""
    spec = dialog["spec"]
    return all(spec.get(k) is not None or gold.get(k) is None for k in ("min_years", "seniority")) \
        and bool(spec.get("title")) and (bool(spec.get("must_have_skills"))
                                         or not gold["must_have_skills"])


def simulate(jd: str) -> dict:
    """Run one dialog; returns history, final spec and turn count."""
    system = llm.load_prompt(SIM).replace("{jd}", " ".join(jd.split()[:900]))
    history: list[dict] = []
    while True:
        # the simulator sees the dialog with roles swapped (it is the "assistant" here)
        sim_msgs = [{"role": "system", "content": system}] + [
            {"role": "assistant" if m["role"] == "user" else "user", "content": m["content"]}
            for m in history
        ] + ([] if history else [{"role": "user", "content": "Start the conversation."}])
        history.append({"role": "user", "content": llm.complete(sim_msgs, component=SIM)})
        turn = next_turn(history)
        history.append({"role": "assistant", "content": turn.reply})
        if turn.done:
            n_user = sum(m["role"] == "user" for m in history)
            return {"history": history, "spec": turn.spec.model_dump(), "turns": n_user,
                    "completed_mandatory": not turn.missing,
                    "hit_turn_limit": n_user >= config.COLLECTOR_MAX_TURNS and bool(turn.missing)}


def aggregate(rows: list[tuple[dict, dict]]) -> dict:
    s = [slot_scores(p, g) for p, g in rows]

    def f1(key: str) -> dict:
        tp, fp, fn = (sum(x[key][i] for x in s) for i in range(3))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}

    return {
        "title_acc": float(np.mean([x["title"] for x in s])),
        "seniority_acc": float(np.mean([x["seniority"] for x in s])),
        "min_years_acc": float(np.mean([x["min_years"] for x in s])),
        "must_have_lexical": f1("must"),
        "nice_to_have_lexical": f1("nice"),
        "must_have_semantic": f1("must_sem"),
        "nice_to_have_semantic": f1("nice_sem"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    sample = [json.loads(x) for x in (GOLD / "jobspec_sample.jsonl").read_text().splitlines()]
    gold = {g["jd_id"]: g for g in map(json.loads, (GOLD / "jobspec_gold.jsonl").read_text().splitlines())}
    sample = sample[: args.limit]

    done = {}
    if OUT.exists():
        done = {r["jd_id"]: r for r in map(json.loads, OUT.read_text().splitlines())}
    dialogs = []
    for s in sample:
        if s["jd_id"] not in done:
            try:
                d = simulate(s["jd_text"])
            except llm.QuotaExhaustedError as e:
                print(f"STOP: {e}")
                break
            done[s["jd_id"]] = {"jd_id": s["jd_id"], **d}
            with open(OUT, "a") as f:
                f.write(json.dumps(done[s["jd_id"]], ensure_ascii=False) + "\n")
        dialogs.append(done[s["jd_id"]])

    full = [(extract_jobspec(s["jd_text"]).model_dump(), gold[s["jd_id"]]) for s in sample]
    dia = [(d["spec"], gold[d["jd_id"]]) for d in dialogs]
    out = {
        "component": "C1 requirements collector",
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {"model": config.LLM_MODEL, "max_turns": config.COLLECTOR_MAX_TURNS,
                   "simulator_prompt": llm.prompt_version(SIM),
                   "extractor_prompt": llm.prompt_version("jobspec_extractor")},
        "n_dialogs": len(dialogs),
        "dialog": {
            **aggregate(dia),
            "completion_rate_mandatory": float(np.mean([d["completed_mandatory"] for d in dialogs])),
            # a mandatory slot left empty is correct when the JD itself does not state it
            "mandatory_resolved_rate": float(np.mean([resolved(d, gold[d["jd_id"]]) for d in dialogs])),
            "mean_turns": float(np.mean([d["turns"] for d in dialogs])),
            "share_within_3_turns": float(np.mean([d["turns"] <= 3 for d in dialogs])),
            "hit_turn_limit": float(np.mean([d["hit_turn_limit"] for d in dialogs])),
        },
        "full_jd_extraction_upper_bound": aggregate(full),
        "llm_calls_used_today": llm.usage_today(config.LLM_MODEL),
    }
    (config.EVAL_DIR / "results" / "collector.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("n_dialogs", "dialog", "full_jd_extraction_upper_bound",
                                          "llm_calls_used_today")}, indent=1))


if __name__ == "__main__":
    main()
