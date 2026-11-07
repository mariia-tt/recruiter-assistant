"""Print the latest LLM calls from traces/ in a compact form (for demos and debugging).

Usage: uv run python scripts/show_traces.py [--n 10] [--full]
"""

import argparse
import json

from recruiter import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="how many latest calls to show")
    parser.add_argument("--full", action="store_true", help="also print prompt and output")
    args = parser.parse_args()

    files = sorted(config.TRACES_DIR.glob("*.jsonl"))
    if not files:
        print("no traces yet")
        return
    records = [json.loads(x) for x in files[-1].read_text().splitlines()][-args.n :]
    for r in records:
        src = "cache" if r["cached"] else f"{r['latency_s']:.1f}s"
        print(f"{r['ts'][11:19]}  {r['component']:<20} {r['model']:<24} {src}")
        if args.full:
            print("  prompt:", r["messages"][-1]["content"][:300].replace("\n", " "), "…")
            print("  output:", r["output"][:300].replace("\n", " "), "…\n")


if __name__ == "__main__":
    main()
