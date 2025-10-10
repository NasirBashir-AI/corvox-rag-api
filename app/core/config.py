# app/core/config.py
"""
Central configuration for Corah.
Reads from environment variables first (so /etc/environment on EC2 wins).
Keeps sensible defaults for local/dev.
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
DB_URL = os.getenv(
    "DB_URL",
    "postgresql://corah_user:password@127.0.0.1:5432/corah"
)

RAW_DOCS_PATH = os.getenv("RAW_DOCS_PATH", "/home/ec2-user/corah/data/raw")

# Embedding model (used by ingestion and retrieval)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


# ---------- Ingestion / chunking ----------
CHUNK_SIZE    = _get_int("CHUNK_SIZE", 900)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 120)


# ---------- RAG / Answer generation ----------
SHOW_CITATIONS  = _get_bool("SHOW_CITATIONS", False)   # include lightweight citations in responses
DEBUG_RAG       = _get_bool("DEBUG_RAG", False)        # include retrieval debug info in API output

# LLM style (warmer but still concise)
TEMPERATURE     = _get_float("TEMPERATURE", 0.2)
MAX_TOKENS      = _get_int("MAX_TOKENS", 700)
MIN_SIM         = _get_float("MIN_SIM", 0.60)
RETRIEVAL_TOP_K = _get_int("RETRIEVAL_TOP_K", 5)

# Query rewrite / self-querying
ENABLE_SELF_QUERY = _get_bool("ENABLE_SELF_QUERY", True)

# Router and retrieval feature flags
ENABLE_SMALLTALK = _get_bool("ENABLE_SMALLTALK", True)
ENABLE_HYBRID    = _get_bool("ENABLE_HYBRID", True)
ENABLE_FACTS     = _get_bool("ENABLE_FACTS", True)

# Lead capture style — “don’t be pushy” tunables
LEAD_NUDGE_COOLDOWN_SEC = _get_int("LEAD_NUDGE_COOLDOWN_SEC", 60)   # minimum gap between nudges
LEAD_MAX_NUDGES         = _get_int("LEAD_MAX_NUDGES", 2)           # per session
ASK_COOLDOWN_SEC        = _get_int("ASK_COOLDOWN_SEC", 45)         # per field ask cooldown (name/phone/email/time/notes)

# Planner/final model names (can be overridden per env)
OPENAI_PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
OPENAI_FINAL_MODEL   = os.getenv("OPENAI_FINAL_MODEL",   os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
OPENAI_EXTRACT_MODEL = os.getenv("OPENAI_EXTRACT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

# Lead summary (optional)
USE_LLM_LEAD_SUMMARY = _get_bool("USE_LLM_LEAD_SUMMARY", True)
LEAD_SUMMARY_MODEL   = os.getenv("LEAD_SUMMARY_MODEL", "gpt-4o-mini")


# ---------- API server defaults ----------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = _get_int("PORT", 8000)


# ---------- convenience echo (optional) ----------
def _mask_db_url(u: str) -> str:
    if "://" not in u or "@" not in u:
        return u
    scheme, rest = u.split("://", 1)
    if "@" in rest:
        _, tail = rest.split("@", 1)
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
    print("[config] LEAD_NUDGE_COOLDOWN_SEC=", LEAD_NUDGE_COOLDOWN_SEC, flush=True)
    print("[config] LEAD_MAX_NUDGES=", LEAD_MAX_NUDGES, flush=True)
    print("[config] ASK_COOLDOWN_SEC=", ASK_COOLDOWN_SEC, flush=True)
    print("[config] OPENAI_PLANNER_MODEL=", OPENAI_PLANNER_MODEL, flush=True)
    print("[config] OPENAI_FINAL_MODEL=", OPENAI_FINAL_MODEL, flush=True)
    print("[config] OPENAI_EXTRACT_MODEL=", OPENAI_EXTRACT_MODEL, flush=True)
    print("[config] USE_LLM_LEAD_SUMMARY=", USE_LLM_LEAD_SUMMARY, flush=True)
    print("[config] LEAD_SUMMARY_MODEL=", LEAD_SUMMARY_MODEL, flush=True)