"""Single entry point for all LLM calls: client, throttling, retries, cache, tracing.

Every component must call `complete` or `structured` from here (AGENTS.md rule 3).
"""

import hashlib
import json
import logging
import re
import threading
import time
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from openai import APIStatusError, OpenAI
from pydantic import BaseModel

from recruiter import config

log = logging.getLogger(__name__)

_client: OpenAI | None = None
_lock = threading.Lock()
# Gemini daily quotas reset at midnight Pacific time.
_QUOTA_TZ = ZoneInfo("America/Los_Angeles")


class QuotaExhaustedError(RuntimeError):
    """Daily request budget is used up — stop instead of hammering the API."""


def _get_client() -> OpenAI:
    """Create the OpenAI-compatible client lazily so tests never need a key.

    SDK-level retries are disabled: every retry must go through our limiter.
    """
    global _client
    if _client is None:
        if not config.LLM_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set — copy .env.example to .env")
        _client = OpenAI(
            api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL, max_retries=0, timeout=90
        )
    return _client


def _limits(model: str) -> tuple[int, int]:
    """(requests per minute, requests per day) configured for a model."""
    if model == config.LLM_MODEL_JUDGE:
        return config.LLM_JUDGE_RPM, config.LLM_JUDGE_RPD
    return config.LLM_RPM, config.LLM_RPD


def _load_usage() -> dict[str, list[float]]:
    try:
        return json.loads(config.USAGE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _quota_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, _QUOTA_TZ).strftime("%Y-%m-%d")


def usage_today(model: str) -> int:
    """Requests sent to `model` in the current quota day (persisted across runs)."""
    today = _quota_day(time.time())
    return sum(_quota_day(t) == today for t in _load_usage().get(model, []))


def _acquire(model: str) -> None:
    """Block until a request fits the sliding 60 s window; fail if the daily budget is spent.

    Every attempt (including ones that end in 429) is recorded, because the
    provider counts them too. Usage is stored on disk so separate script runs
    share the same budget.
    """
    rpm, rpd = _limits(model)
    with _lock:
        usage = _load_usage()
        now = time.time()
        stamps = [t for t in usage.get(model, []) if now - t < 26 * 3600]
        today = _quota_day(now)
        if sum(_quota_day(t) == today for t in stamps) >= rpd:
            raise QuotaExhaustedError(f"{model}: daily budget of {rpd} requests used up")
        window = [t for t in stamps if now - t < 60]
        if len(window) >= rpm:
            wait = 60 - (now - window[-rpm]) + 0.5
            log.info("rate limit: waiting %.1fs for %s", wait, model)
            time.sleep(wait)
            now = time.time()
        stamps.append(now)
        usage[model] = stamps
        config.USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.USAGE_FILE.write_text(json.dumps(usage))


def _retry_delay(err: APIStatusError) -> float | None:
    """Server-suggested delay (Gemini RetryInfo, e.g. "17s"), if present."""
    match = re.search(r"retry[_ ]?delay\W+(\d+(?:\.\d+)?)s", str(err.body), re.IGNORECASE)
    return float(match.group(1)) if match else None


def _is_daily_quota(err: APIStatusError) -> bool:
    body = str(err.body)
    return "PerDay" in body or "per day" in body.lower()


def _cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _trace(component: str, payload: dict, output: str, latency: float, cached: bool) -> None:
    """Append one LLM call to traces/<date>.jsonl for debugging and the demo."""
    config.TRACES_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    record = {
        "ts": now.isoformat(),
        "component": component,
        "model": payload["model"],
        "cached": cached,
        "latency_s": round(latency, 2),
        "messages": payload["messages"],
        "output": output,
    }
    with open(config.TRACES_DIR / f"{now:%Y-%m-%d}.jsonl", "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _call(payload: dict, component: str, max_retries: int = 3) -> str:
    """Chat completion with disk cache; limited retries on per-minute 429 and 5xx."""
    key = _cache_key(payload)
    cache_file = config.CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        output = json.loads(cache_file.read_text())["output"]
        _trace(component, payload, output, 0.0, cached=True)
        return output

    model = payload["model"]
    for attempt in range(max_retries + 1):
        _acquire(model)
        start = time.monotonic()
        try:
            resp = _get_client().chat.completions.create(**payload)
            output = resp.choices[0].message.content or ""
            break
        except APIStatusError as e:
            if e.status_code == 429 and _is_daily_quota(e):
                raise QuotaExhaustedError(f"{model}: provider reports daily quota exhausted") from e
            retryable = e.status_code == 429 or e.status_code >= 500
            if not retryable or attempt == max_retries:
                raise
            delay = _retry_delay(e) or 30 * 2**attempt
            log.warning("LLM %s on %s, retry %d/%d in %.0fs",
                        e.status_code, model, attempt + 1, max_retries, delay)
            time.sleep(delay)

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"payload": payload, "output": output}, ensure_ascii=False))
    _trace(component, payload, output, time.monotonic() - start, cached=False)
    return output


def complete(
    messages: list[dict], *, component: str, model: str | None = None, temperature: float = 0.0
) -> str:
    """Free-text completion (used for dialog turns and outreach messages)."""
    payload = {"model": model or config.LLM_MODEL, "messages": messages, "temperature": temperature}
    return _call(payload, component)


def structured[T: BaseModel](
    messages: list[dict],
    schema: type[T],
    *,
    component: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> T:
    """Completion parsed into a Pydantic model via JSON-schema constrained output."""
    payload = {
        "model": model or config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        },
    }
    return schema.model_validate_json(_call(payload, component))


def load_prompt(name: str) -> str:
    """Read prompts/<name>.md without its first `version:` line."""
    text = (config.PROMPTS_DIR / f"{name}.md").read_text()
    return text.split("\n", 1)[1].lstrip() if text.startswith("version:") else text


def prompt_version(name: str) -> str:
    """Return the version tag of a prompt, stored in eval results for traceability."""
    first = (config.PROMPTS_DIR / f"{name}.md").read_text().split("\n", 1)[0]
    return first.removeprefix("version:").strip()
