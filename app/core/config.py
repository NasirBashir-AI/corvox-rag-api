# app/core/config.py
from __future__ import annotations
import os

def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in {"1","true","yes","y","on"}

def _get_int(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except Exception: return default

def _get_float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except Exception: return default

# Core
DB_URL = os.getenv("DB_URL", "postgresql://corah_user:password@127.0.0.1:5432/corah")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# RAG
RETRIEVAL_TOP_K = _get_int("RETRIEVAL_TOP_K", 5)

# Temperatures
PLANNER_TEMPERATURE = _get_float("PLANNER_TEMPERATURE", 0.3)
FINAL_TEMPERATURE   = _get_float("FINAL_TEMPERATURE", 0.5)
TEMPERATURE         = _get_float("TEMPERATURE", 0.5)  # legacy fallback

# Conversation policy
CTA_COOLDOWN_TURNS    = _get_int("CTA_COOLDOWN_TURNS", 3)
CTA_MAX_ATTEMPTS      = _get_int("CTA_MAX_ATTEMPTS", 2)
ONE_QUESTION_MAX      = _get_bool("ONE_QUESTION_MAX", True)
ASK_EARLY             = _get_bool("ASK_EARLY", False)  # if False, don’t ask in first 2 assistant turns
ASK_MIN_TURN_INDEX    = _get_int("ASK_MIN_TURN_INDEX", 3)  # earliest overall turn to start asking
ALLOW_PUBLIC_CONTACT  = _get_bool("ALLOW_PUBLIC_CONTACT", True)  # OK to share email/address if retrieved

# Inactivity (frontend hints only; backend may use separately)
INACTIVITY_MINUTES = _get_int("CORAH_INACTIVITY_MINUTES", 5)