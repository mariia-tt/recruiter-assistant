"""Validate the C6 grounding checker (an LLM judge) against independent claim labels.

Gold: eval/gold/claims_gold.jsonl (61 claims from the outreach eval, independent LLM
annotator that did not see C6 verdicts). C6 verdicts: eval/results/c6_verdicts.json.
No LLM calls.
"""

import json
from datetime import UTC, datetime

from recruiter import config


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Agreement beyond chance for two binary raters."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    gold = [json.loads(x)["supported"] for x in
            (config.EVAL_DIR / "gold" / "claims_gold.jsonl").read_text().splitlines()]
    c6 = json.loads((config.EVAL_DIR / "results" / "c6_verdicts.json").read_text())
    # positive class = "unsupported" (what the guardrail must catch)
    tp = sum(not g and not p for g, p in zip(gold, c6, strict=True))
    fp = sum(g and not p for g, p in zip(gold, c6, strict=True))
    fn = sum(not g and p for g, p in zip(gold, c6, strict=True))
    out = {
        "component": "C6 grounding checker (judge validation)",
        "timestamp": datetime.now(UTC).isoformat(),
        "n_claims": len(gold),
        "gold_unsupported": sum(not g for g in gold),
        "c6_flagged": sum(not p for p in c6),
        "accuracy": sum(g == p for g, p in zip(gold, c6, strict=True)) / len(gold),
        "unsupported_precision": tp / (tp + fp) if tp + fp else None,
        "unsupported_recall": tp / (tp + fn) if tp + fn else None,
        "cohen_kappa": cohen_kappa(gold, c6),
        "confusion": {"both_unsupported": tp, "c6_only_flagged": fp, "c6_missed": fn,
                      "both_supported": len(gold) - tp - fp - fn},
    }
    (config.EVAL_DIR / "results" / "grounding_judge.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
