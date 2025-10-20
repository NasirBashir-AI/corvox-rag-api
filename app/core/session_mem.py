# app/core/session_mem.py
"""
Lightweight in-memory session manager for Corah.
Persists recent turns, summary, and lead slots per session.
Phase 2.1: adds slot authority, CTA cooldown, and safe resets.
"""

from __future__ import annotations
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

# ---------------------------------------------------------------------
# In-memory store (temporary; in production this can be Redis)
# ---------------------------------------------------------------------
_SESSIONS: Dict[str, Dict[str, Any]] = {}

TURN_MEMORY_LIMIT = 10
CTA_COOLDOWN_TURNS = 2     # at least 2 assistant turns before next CTA
SESSION_TTL_MINUTES = 30   # stale cleanup window

# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_state(session_id: str) -> Optional[Dict[str, Any]]:
    """Return current session dict."""
    return _SESSIONS.get(session_id)

def set_state(session_id: str, **fields: Any) -> None:
    """Update or create session dict with given fields."""
    state = _SESSIONS.setdefault(session_id, {})
    state.update(fields)
    state["updated_at"] = _now_iso()

def clear_state(session_id: str) -> None:
    """Wipe a session completely."""
    if session_id in _SESSIONS:
        del _SESSIONS[session_id]

# ---------------------------------------------------------------------
# Turn management
# ---------------------------------------------------------------------
def append_turn(session_id: str, role: str, content: str) -> None:
    """Append a message turn (user/assistant)."""
    state = _SESSIONS.setdefault(session_id, {})
    turns = state.setdefault("turns", [])
    turns.append({"role": role, "content": content, "ts": _now_iso()})
    if len(turns) > TURN_MEMORY_LIMIT:
        turns[:] = turns[-TURN_MEMORY_LIMIT:]
    state["turns_count"] = state.get("turns_count", 0) + 1
    state["updated_at"] = _now_iso()

def recent_turns(session_id: str, n: int = 6) -> List[Dict[str, Any]]:
    state = _SESSIONS.get(session_id, {})
    return state.get("turns", [])[-n:]

# ---------------------------------------------------------------------
# Summary + topic tracking
# ---------------------------------------------------------------------
def update_summary(session_id: str) -> None:
    """Light heuristic summary (can later be LLM-generated)."""
    st = _SESSIONS.setdefault(session_id, {})
    turns = st.get("turns", [])
    if not turns:
        return
    # take last user message for quick summary
    last_user = next((t["content"] for t in reversed(turns) if t["role"] == "user"), "")
    st["session_summary"] = (last_user[:150] + "...") if last_user else "-"
    st["current_topic"] = st.get("current_topic") or "general"
    st["updated_at"] = _now_iso()

# ---------------------------------------------------------------------
# Lead slot management (Phase 2.1)
# ---------------------------------------------------------------------
def get_lead_slots(session_id: str) -> Dict[str, Any]:
    st = _SESSIONS.setdefault(session_id, {})
    return st.setdefault("lead", {
        "name": None,
        "company": None,
        "email": None,
        "phone": None,
        "preferred_time": None,
        "notes": None,
    })

def update_lead_slot(session_id: str, field: str, value: str) -> None:
    st = _SESSIONS.setdefault(session_id, {})
    lead = get_lead_slots(session_id)
    if value and (lead.get(field) != value):
        lead[field] = value.strip()
        st["last_slot_updated"] = field
        st["updated_at"] = _now_iso()

def all_lead_info_complete(session_id: str) -> bool:
    lead = get_lead_slots(session_id)
    required = ["name", "company"]
    has_contact = bool(lead.get("email") or lead.get("phone"))
    return all(lead.get(f) for f in required) and has_contact

# ---------------------------------------------------------------------
# CTA / cooldown logic
# ---------------------------------------------------------------------
def can_offer_cta(session_id: str) -> bool:
    """Return True if enough turns have passed since last CTA."""
    st = _SESSIONS.get(session_id, {})
    turns_since = (st.get("turns_count", 0) - st.get("cta_last_turn", 0))
    return turns_since >= CTA_COOLDOWN_TURNS

def mark_cta_used(session_id: str) -> None:
    st = _SESSIONS.setdefault(session_id, {})
    st["cta_last_turn"] = st.get("turns_count", 0)
    st["cta_attempts"] = st.get("cta_attempts", 0) + 1
    st["updated_at"] = _now_iso()

def reset_cta_cooldown(session_id: str) -> None:
    st = _SESSIONS.setdefault(session_id, {})
    st["cta_last_turn"] = 0
    st["cta_attempts"] = 0

# ---------------------------------------------------------------------
# Session lifecycle (closure + expiry)
# ---------------------------------------------------------------------
def mark_closed(session_id: str) -> None:
    """Mark session closed (for polite exit)."""
    st = _SESSIONS.setdefault(session_id, {})
    st["closed_at"] = _now_iso()
    st["is_closed"] = True
    st["updated_at"] = _now_iso()

def cleanup_expired() -> None:
    """Remove old sessions beyond TTL."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=SESSION_TTL_MINUTES)
    stale = [sid for sid, s in _SESSIONS.items()
             if datetime.fromisoformat(s.get("updated_at", _now_iso())) < cutoff]
    for sid in stale:
        del _SESSIONS[sid]