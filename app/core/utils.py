"""
app/core/utils.py

Shared, minimal helpers used across Corah.
- Supports psycopg (v3) or psycopg2 for Postgres access
- Small, stable utilities reused across modules
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence


# -----------------------------
# Environment helpers
# -----------------------------

def getenv_str(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get env var as string; returns default if missing/empty."""
    val = os.getenv(name)
    if val is None or str(val).strip() == "":
        return default
    return val


def getenv_bool(name: str, default: bool = False) -> bool:
    """Get env var as boolean with common truthy values."""
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def getenv_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def getenv_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def mask_db_url(url: str) -> str:
    """Hide credentials in a DB URL for safe logging."""
    if not url or "://" not in url or "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if "@" in rest:
        _creds, tail = rest.split("@", 1)
        return f"{scheme}://***:***@{tail}"
    return url


# -----------------------------
# Text helpers
# -----------------------------

_WS_COLLAPSE_RE = re.compile(r"\s+")
_FILE_TOKEN_RE = re.compile(r"\b[\w\-/]+\.(md|pdf|txt|docx)\b", re.IGNORECASE)
_BRACKETED_SOURCE_RE = re.compile(r"\s*[\(\[]\s*(source|file|doc|kb)[:=].*?[\)\]]", re.IGNORECASE)
_HDR_LINE_RE = re.compile(r"^\s*\[[^\]]+\]\s*", re.MULTILINE)  # strips lines starting with [ ... ]
_SOURCE_LINE_RE = re.compile(r"^\s*(source|file|doc|kb)\s*[:=].*$", re.IGNORECASE | re.MULTILINE)

def normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to single spaces and trim."""
    return _WS_COLLAPSE_RE.sub(" ", text or "").strip()


def strip_source_tokens(text: str) -> str:
    """
    Remove bracketed context headers like `[snippet · #2]`, file-name tokens (e.g., `contact.md`),
    and bracketed/inline 'source:' notes from model outputs so user-facing text stays clean.
    """
    if not text:
        return text
    t = text
    t = _HDR_LINE_RE.sub("", t)
    t = _FILE_TOKEN_RE.sub("", t)
    t = _BRACKETED_SOURCE_RE.sub("", t)
    t = _SOURCE_LINE_RE.sub("", t)
    return normalize_ws(t)


def truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars with a neat ellipsis."""
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


# -----------------------------
# Postgres helpers (psycopg / psycopg2)
# -----------------------------

# Try psycopg (v3) first, then fall back to psycopg2
_psycopg = None
_psycopg2 = None
try:  # psycopg v3
    import psycopg  # type: ignore
    _psycopg = psycopg
except Exception:
    _psycopg = None

if _psycopg is None:
    try:  # psycopg2 fallback
        import psycopg2  # type: ignore
        _psycopg2 = psycopg2
    except Exception:
        _psycopg2 = None


def _ensure_pg_driver_available() -> None:
    if _psycopg is None and _psycopg2 is None:
        raise RuntimeError(
            "No Postgres driver found. Install either 'psycopg[binary]' (v3) or 'psycopg2-binary'."
        )


def get_pg_conn(db_url: Optional[str] = None):
    """
    Return a live Postgres connection using either psycopg (v3) or psycopg2.
    Caller is responsible for closing the connection, or use the context managers below.
    """
    _ensure_pg_driver_available()
    url = db_url or getenv_str("DB_URL")
    if not url:
        raise RuntimeError("DB_URL is not set in the environment.")

    if _psycopg is not None:
        return _psycopg.connect(url)
    return _psycopg2.connect(url)  # type: ignore[attr-defined]


@contextmanager
def pg_conn(db_url: Optional[str] = None):
    """Context manager that yields a Postgres connection and ensures it gets closed."""
    conn = get_pg_conn(db_url)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


@contextmanager
def pg_cursor(db_url: Optional[str] = None):
    """
    Context manager that yields a cursor and commits on success.
    Works with both psycopg (v3) and psycopg2.
    """
    with pg_conn(db_url) as conn:
        cur = conn.cursor()
        try:
            yield cur
            try:
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
        finally:
            try:
                cur.close()
            except Exception:
                pass


def rows_to_dicts(cur) -> List[Dict[str, Any]]:
    """
    Convert the current cursor result set to a list of dicts.
    Safe for psycopg (v3) and psycopg2: gracefully handles different .description shapes.
    """
    desc = getattr(cur, "description", None)
    if not desc:
        return []
    columns: List[str] = []
    for idx, col in enumerate(desc):
        # psycopg3 exposes objects with .name; psycopg2 gives a sequence (name at index 0)
        name = getattr(col, "name", None)
        if name:
            columns.append(str(name))
        elif isinstance(col, (list, tuple)) and len(col) > 0:
            columns.append(str(col[0]))
        else:
            columns.append(f"col_{idx}")
    return [dict(zip(columns, row)) for row in cur.fetchall()]


# -----------------------------
# Small safe utilities used across modules
# -----------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def soft_normalize(scores: Sequence[float]) -> List[float]:
    """
    Normalize a list of scores into 0..1 range.
    If all values are equal, return 1.0 for each (avoids zeroed signals).
    """
    if not scores:
        return []
    vals = [float(s) for s in scores]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [1.0 for _ in vals]
    span = hi - lo
    return [(s - lo) / span for s in vals]