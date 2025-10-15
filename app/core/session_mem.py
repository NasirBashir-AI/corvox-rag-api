# app/core/session_mem.py
"""
Lightweight in-memory session state for Corah.

Responsibilities
- Create/ensure a session id
- Store ephemeral per-session fields (name, email, phone, company, etc.)
- Track last activity timestamps for inactivity handling
- Maintain a minimal rolling summary for lead notes
- Provide clear() to wipe state on close/timeout

Notes
- This is intentionally in-memory. For multi-instance deployments,
  swap the backend with Redis or Postgres but keep the same function contracts.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import threading
import uuid

# ---------------------------
# Internal store
# ---------------------------

_LOCK = threading.Lock()
_STORE: Dict[str, Dict[str, Any]] = {}

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ---------------------------
# Public API
# ---------------------------

def ensure_session(session_id: Optional[str]) -> str:
    """
    Ensure a session exists; return the active session_id.
    If session_id is None or unknown, create a new one.
    """
    global _STORE
    if not session_id:
        sid = str(uuid.uuid4())
        with _LOCK:
            _STORE[sid] = {"created_at": _now_iso(), "last_active_at": _now_iso(), "turns": []}
        return sid

    with _LOCK:
        if session_id not in _STORE:
            _STORE[session_id] = {"created_at": _now_iso(), "last_active_at": _now_iso(), "turns": []}
    return session_id


def get_state(session_id: str) -> Dict[str, Any]:
    """Return the session dict; empty dict if not present."""
    with _LOCK:
        return dict(_STORE.get(session_id, {}))


def set_state(session_id: str, **fields: Any) -> None:
    """Merge the provided fields into the session dict."""
    with _LOCK:
        st = _STORE.setdefault(session_id, {"created_at": _now_iso(), "turns": []})
        st.update(fields)
        # Keep a tiny rolling trail of the last user text, if provided
        if "last_user_text" in fields and fields["last_user_text"]:
            turns: List[Dict[str, Any]] = st.setdefault("turns", [])
            turns.append({"role": "user", "content": fields["last_user_text"], "ts": _now_iso()})
            # cap to last 6 to avoid growth
            if len(turns) > 6:
                del turns[:-6]
        st["last_active_at"] = _now_iso()


def touch_session(session_id: str) -> None:
    """Update last_active_at for the session."""
    with _LOCK:
        st = _STORE.setdefault(session_id, {"created_at": _now_iso(), "turns": []})
        st["last_active_at"] = _now_iso()


def summarize_session(session_id: str) -> str:
    """
    Produce a short, human-friendly line for lead notes.
    Uses available fields without calling external models.
    """
    with _LOCK:
        st = _STORE.get(session_id, {})
        if not st:
            return ""

        name = st.get("name") or "—"
        company = st.get("company") or "—"
        interest = st.get("last_interest_level") or "none"
        priority = st.get("last_priority") or "cold"
        # last two user turns for context
        turns = st.get("turns") or []
        last_bits = [t.get("content", "") for t in turns[-2:]] if turns else []
        last_snippet = " | ".join(b for b in last_bits if b).strip()
        if len(last_snippet) > 220:
            last_snippet = last_snippet[:217] + "..."

        summary = f"Name={name}; Company={company}; Interest={interest}; Priority={priority}"
        if last_snippet:
            summary += f"; Recent: {last_snippet}"
        st["session_summary"] = summary
        return summary


def clear_session(session_id: str) -> None:
    """Remove the session state entirely (called on user_end/timeout)."""
    with _LOCK:
        if session_id in _STORE:
            del _STORE[session_id]