from __future__ import annotations
import os

def _b(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else str(v).strip().lower() in {"1","true","yes","y","on"}

def _i(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except: return default

def _f(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except: return default

# --- Core ---
DB_URL = os.getenv("DB_URL", "postgresql://corah_user:password@127.0.0.1:5432/corah")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
RETRIEVAL_TOP_K = _i("RETRIEVAL_TOP_K", 5)

# --- Behavior flags ---
ANSWER_ONLY_UNTIL_INTENT_TURNS = _i("ANSWER_ONLY_UNTIL_INTENT_TURNS", 2)
CTA_COOLDOWN_TURNS             = _i("CTA_COOLDOWN_TURNS", 2)
CTA_MAX_ATTEMPTS               = _i("CTA_MAX_ATTEMPTS", 2)

# Lead requirements (soft; we won’t nag)
REQUIRE_COMPANY        = _b("REQUIRE_COMPANY", True)
REQUIRE_EMAIL_OR_PHONE = _b("REQUIRE_EMAIL_OR_PHONE", True)

# Debug surfacing
DEBUG_RAG = _b("DEBUG_RAG", False)
SHOW_CITATIONS = _b("SHOW_CITATIONS", True)