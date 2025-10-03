"""
Central config for Corah.
- Reads from environment first (so /etc/environment on EC2 wins)
- Falls back to sensible defaults for local dev
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
# OpenAI API key is read implicitly by the OpenAI SDK from the environment:
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

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ---------- Ingestion / chunking (NEW) ----------
# Used by the ingestion pipeline and sometimes by retrieval for windowing
CHUNK_SIZE    = _get_int("CHUNK_SIZE", 900)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 120)

# ---------- RAG answer controls ----------
# Surface-level flags (you can toggle at runtime via env without code changes)
SHOW_CITATIONS = _get_bool("SHOW_CITATIONS", False)  # include lightweight citations in responses
DEBUG_RAG      = _get_bool("DEBUG_RAG", False)       # include retrieval debug info in API output

# LLM style controls
TEMPERATURE    = _get_float("TEMPERATURE", 0.30)     # 0.2–0.5 recommended
MAX_TOKENS     = _get_int("MAX_TOKENS", 600)         # answer token cap (used inside generator)

# Retrieval quality gate
MIN_SIM            = _get_float("MIN_SIM", 0.60)     # similarity threshold (0..1)
ENABLE_SELF_QUERY  = _get_bool("ENABLE_SELF_QUERY", True)  # turn self-query rewriting on/off

# ---------- Lead capture / reporting ----------
USE_LLM_LEAD_SUMMARY = _get_bool("USE_LLM_LEAD_SUMMARY", True)
LEAD_SUMMARY_MODEL   = os.getenv("LEAD_SUMMARY_MODEL", "gpt-4o-mini")

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
    print("[config] ENABLE_SELF_QUERY=", ENABLE_SELF_QUERY, flush=True)
    print("[config] USE_LLM_LEAD_SUMMARY=", USE_LLM_LEAD_SUMMARY, flush=True)
    print("[config] LEAD_SUMMARY_MODEL=", LEAD_SUMMARY_MODEL, flush=True)
    print("[config] HOST=", HOST, "PORT=", PORT, flush=True)