"""
app/core/config.py

Central configuration for Corah.
- Reads from environment variables first (so /etc/environment on EC2 wins)
- Falls back to sensible defaults for local/dev
- Keep this file self-contained (no imports from other app modules to avoid cycles)
"""

from __future__ import annotations
import os

# ---------- helpers ----------
def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")

def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


# ---------- core services ----------
# OpenAI API key is read implicitly by the OpenAI SDK:
#   export OPENAI_API_KEY=sk-...
# No need to duplicate it here.

# Postgres connection (prefer DB_URL if provided)
DB_URL = os.getenv(
    "DB_URL",
    # local/dev default; override in /etc/environment on EC2 with your RDS URL
    "postgresql://corah_user:password@127.0.0.1:5432/corah"
)

# Where raw knowledge files live on the server
RAW_DOCS_PATH = os.getenv(
    "RAW_DOCS_PATH",
    "/home/ec2-user/corah/data/raw"
)

# Embedding model (used by ingestion and retrieval)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


# ---------- Ingestion / chunking ----------
CHUNK_SIZE    = _get_int("CHUNK_SIZE", 900)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 120)


# ---------- RAG / Answer generation controls ----------
# Surface flags (toggleable at runtime via env without code changes)
SHOW_CITATIONS  = _get_bool("SHOW_CITATIONS", False)   # include lightweight citations in responses
DEBUG_RAG       = _get_bool("DEBUG_RAG", False)        # include retrieval debug info in API output

# LLM style controls (Phase A: more deterministic, concise)
TEMPERATURE     = _get_float("TEMPERATURE", 0.10)      # tightened for crisp, grounded answers
MAX_TOKENS      = _get_int("MAX_TOKENS", 600)          # cap for completion tokens
MIN_SIM         = _get_float("MIN_SIM", 0.60)          # similarity threshold (0..1) used for diagnostics
RETRIEVAL_TOP_K = _get_int("RETRIEVAL_TOP_K", 5)       # default top-k for retrieval when not specified

# Query rewrite / self-querying
ENABLE_SELF_QUERY = _get_bool("ENABLE_SELF_QUERY", True)

# Router and retrieval feature flags (legacy toggles; safe to ignore if unused)
ENABLE_SMALLTALK = _get_bool("ENABLE_SMALLTALK", True) # legacy smalltalk bypass
ENABLE_HYBRID    = _get_bool("ENABLE_HYBRID", True)    # vector + FTS blending in Postgres
ENABLE_FACTS     = _get_bool("ENABLE_FACTS", True)     # use structured facts for contact/pricing if available

# Lead capture / reporting
USE_LLM_LEAD_SUMMARY = _get_bool("USE_LLM_LEAD_SUMMARY", True)
LEAD_SUMMARY_MODEL   = os.getenv("LEAD_SUMMARY_MODEL", "gpt-4o-mini")

# ---------- Lead flow guardrails & cooldowns ----------
# Global nudger cooldown and budget
LEAD_NUDGE_COOLDOWN_SEC = _get_int("LEAD_NUDGE_COOLDOWN_SEC", 60)
LEAD_MAX_NUDGES         = _get_int("LEAD_MAX_NUDGES", 2)

# Per-field "don't repeat" cool-downs (fallback to ASK_COOLDOWN_SEC)
ASK_COOLDOWN_SEC        = _get_int("ASK_COOLDOWN_SEC", 45)
ASK_COOLDOWN_NAME_SEC   = _get_int("ASK_COOLDOWN_NAME_SEC", ASK_COOLDOWN_SEC)
ASK_COOLDOWN_PHONE_SEC  = _get_int("ASK_COOLDOWN_PHONE_SEC", ASK_COOLDOWN_SEC)
ASK_COOLDOWN_EMAIL_SEC  = _get_int("ASK_COOLDOWN_EMAIL_SEC", ASK_COOLDOWN_SEC)
ASK_COOLDOWN_TIME_SEC   = _get_int("ASK_COOLDOWN_TIME_SEC", ASK_COOLDOWN_SEC)
ASK_COOLDOWN_NOTES_SEC  = _get_int("ASK_COOLDOWN_NOTES_SEC", ASK_COOLDOWN_SEC)

# LLM fallback extractor thresholds
NAME_LLM_MIN_CONF       = _get_float("NAME_LLM_MIN_CONF", 0.92)


# ---------- API server defaults (used by run scripts, not FastAPI itself) ----------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = _get_int("PORT", 8000)


# ---------- convenience echo (optional) ----------
def _mask_db_url(u: str) -> str:
    if "://" not in u or "@" not in u:
        return u
    scheme, rest = u.split("://", 1)
    if "@" in rest:
        creds, tail = rest.split("@", 1)
        return f"{scheme}://***:***@{tail}"
    return u

if _get_bool("CONFIG_ECHO", False):
    print("[config] DB_URL=", _mask_db_url(DB_URL), flush=True)
    print("[config] RAW_DOCS_PATH=", RAW_DOCS_PATH, flush=True)
    print("[config] EMBEDDING_MODEL=", EMBEDDING_MODEL, flush=True)
    print("[config] CHUNK_SIZE=", CHUNK_SIZE, flush=True)
    print("[config] CHUNK_OVERLAP=", CHUNK_OVERLAP, flush=True)
    print("[config] SHOW_CITATIONS=", SHOW_CITATIONS, flush=True)
    print("[config] DEBUG_RAG=", DEBUG_RAG, flush=True)
    print("[config] TEMPERATURE=", TEMPERATURE, flush=True)
    print("[config] MAX_TOKENS=", MAX_TOKENS, flush=True)
    print("[config] MIN_SIM=", MIN_SIM, flush=True)
    print("[config] RETRIEVAL_TOP_K=", RETRIEVAL_TOP_K, flush=True)
    print("[config] ENABLE_SELF_QUERY=", ENABLE_SELF_QUERY, flush=True)
    print("[config] ENABLE_SMALLTALK=", ENABLE_SMALLTALK, flush=True)
    print("[config] ENABLE_HYBRID=", ENABLE_HYBRID, flush=True)
    print("[config] ENABLE_FACTS=", ENABLE_FACTS, flush=True)
    print("[config] USE_LLM_LEAD_SUMMARY=", USE_LLM_LEAD_SUMMARY, flush=True)
    print("[config] LEAD_SUMMARY_MODEL=", LEAD_SUMMARY_MODEL, flush=True)
    print("[config] LEAD_NUDGE_COOLDOWN_SEC=", LEAD_NUDGE_COOLDOWN_SEC, flush=True)
    print("[config] LEAD_MAX_NUDGES=", LEAD_MAX_NUDGES, flush=True)
    print("[config] ASK_COOLDOWN_SEC=", ASK_COOLDOWN_SEC, flush=True)
    print("[config] ASK_COOLDOWN_NAME_SEC=", ASK_COOLDOWN_NAME_SEC, flush=True)
    print("[config] ASK_COOLDOWN_PHONE_SEC=", ASK_COOLDOWN_PHONE_SEC, flush=True)
    print("[config] ASK_COOLDOWN_EMAIL_SEC=", ASK_COOLDOWN_EMAIL_SEC, flush=True)
    print("[config] ASK_COOLDOWN_TIME_SEC=", ASK_COOLDOWN_TIME_SEC, flush=True)
    print("[config] ASK_COOLDOWN_NOTES_SEC=", ASK_COOLDOWN_NOTES_SEC, flush=True)
    print("[config] NAME_LLM_MIN_CONF=", NAME_LLM_MIN_CONF, flush=True)