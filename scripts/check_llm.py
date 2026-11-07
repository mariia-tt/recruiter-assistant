"""Smoke test: ONE request per configured model, then print today's usage."""

from openai import APIError
from pydantic import BaseModel, ValidationError

from recruiter import config, llm


class Ping(BaseModel):
    ok: bool
    model_said: str


def main() -> None:
    for model in (config.LLM_MODEL, config.LLM_MODEL_JUDGE):
        msgs = [{"role": "user", "content": "Return ok=true and model_said='hello'."}]
        try:
            print(model, "->", llm.structured(msgs, Ping, component="smoke_test", model=model))
        except (APIError, ValidationError, llm.QuotaExhaustedError) as e:  # report, never retry here
            print(model, "-> FAILED:", type(e).__name__, str(e)[:200])
        print(f"   used today: {llm.usage_today(model)}")


if __name__ == "__main__":
    main()
