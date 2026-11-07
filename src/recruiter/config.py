"""Central configuration: paths, model names and business-logic weights."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CACHE_DIR = ROOT / "data" / "cache"
TRACES_DIR = ROOT / "traces"
PROMPTS_DIR = ROOT / "prompts"
EVAL_DIR = ROOT / "eval"

LLM_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
# Workhorse model for every pipeline component (highest free-tier quota).
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
# Stronger model used only as an independent judge on small eval samples.
LLM_MODEL_JUDGE = os.getenv("LLM_MODEL_JUDGE", "gemini-3.8-flash")
# Client-side limits, set slightly BELOW the provider's free-tier limits.
LLM_RPM = int(os.getenv("LLM_RPM", "12"))
LLM_RPD = int(os.getenv("LLM_RPD", "450"))
LLM_JUDGE_RPM = int(os.getenv("LLM_JUDGE_RPM", "4"))
LLM_JUDGE_RPD = int(os.getenv("LLM_JUDGE_RPD", "18"))
USAGE_FILE = CACHE_DIR / "_usage.json"

# Data preparation
MIN_CV_WORDS = 50
LLM_MAX_CV_WORDS = 2000
LLM_MAX_JD_WORDS = 1500
# Retrieval (C3)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_WORDS = 200
CHUNK_OVERLAP = 40
RRF_K = 60
RETRIEVE_K = 50

# CVs were scraped ~2019–2020; "Current"/"Present" in a date range means this year.
DATA_REFERENCE_YEAR = 2020

# Scorecard ranker (C4). v1 = business weights from docs/01_business_understanding.md;
# v2 = revised after evaluation (see docs/05_evaluation.md, CRISP-DM log 2026-11-07).
SCORECARD_WEIGHTS_V1 = {
    "must_have": 0.40, "experience": 0.20, "title": 0.15,
    "nice_to_have": 0.10, "seniority": 0.10, "domain": 0.05, "semantic": 0.0,
}
SCORECARD_WEIGHTS = {
    "must_have": 0.25, "experience": 0.25, "title": 0.10,
    "nice_to_have": 0.05, "seniority": 0.05, "domain": 0.10, "semantic": 0.20,
}
# Hard knock-out on must-have coverage. Disabled by default: on the benchmark it rejects
# Good Fit almost as often as No Fit. None = off; missing must-haves are shown as a flag.
KNOCKOUT_MUST_COVERAGE: float | None = None
# bge cosine similarities live roughly in [0.5, 0.9]; rescale to [0, 1].
SIM_LOW, SIM_HIGH = 0.5, 0.9
RERANK_TOP_N = 10
# LLM re-ordering did not beat scorecard v2 (NDCG −0.009, n.s.): use LLM for explanations only.
RERANK_REORDER = False
RERANK_CV_WORDS = 450

# Requirements collector (C1): after this many recruiter turns, search with what we have.
COLLECTOR_MAX_TURNS = 4

# Outreach (C5) and grounding checker (C6)
OUTREACH_MAX_WORDS = 120
OUTREACH_MAX_ATTEMPTS = 2
