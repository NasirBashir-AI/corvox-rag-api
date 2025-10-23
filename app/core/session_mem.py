# app/core/session_mem.py
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

_SESSIONS: Dict[str, Dict[str, Any]] = {}
TURN_MEMORY_LIMIT = 10
CTA_COOLDOWN_TURNS = 2
SESSION_TTL_MINUTES = 30

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_state(session_id: str) -> Optional[Dict[str, Any]]:
    return _SESSIONS.get(session_id)

def set_state(session_id: str, **fields: Any) -> None:
    st = _SESSIONS.setdefault(session_id, {})
    st.update(fields); st["updated_at"] = _now_iso()

def clear_state(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)

def append_turn(session_id: str, role: str, content: str) -> None:
    st = _SESSIONS.setdefault(session_id, {})
    turns = st.setdefault("turns", [])
    turns.append({"role": role, "content": content, "ts": _now_iso()})
    if len(turns) > TURN_MEMORY_LIMIT:
        turns[:] = turns[-TURN_MEMORY_LIMIT:]
    st["turns_count"] = st.get("turns_count", 0) + 1
    st["updated_at"] = _now_iso()

def recent_turns(session_id: str, n: int = 6) -> List[Dict[str, Any]]:
    return (_SESSIONS.get(session_id, {}).get("turns") or [])[-n:]

def update_summary(session_id: str) -> None:
    st = _SESSIONS.setdefault(session_id, {})
    turns = st.get("turns", [])
    if not turns: return
    last_user = next((t["content"] for t in reversed(turns) if t["role"]=="user"), "")
    quick = (last_user[:150] + "...") if last_user and len(last_user) > 150 else (last_user or "-")
    st["summary"] = quick
    st["session_summary"] = quick
    st["current_topic"] = st.get("current_topic") or "general"
    st["updated_at"] = _now_iso()

def get_lead_slots(session_id: str) -> Dict[str, Any]:
    st = _SESSIONS.setdefault(session_id, {})
    return st.setdefault("lead", {
        "name": None, "company": None, "email": None, "phone": None,
        "preferred_time": None, "notes": None
    })

def update_lead_slot(session_id: str, field: str, value: Optional[str]) -> None:
    if not value: return
    v = value.strip()
    if not v: return
    st = _SESSIONS.setdefault(session_id, {})
    lead = get_lead_slots(session_id)
    if lead.get(field) != v:
        lead[field] = v
        st["last_slot_updated"] = field
        st["updated_at"] = _now_iso()

def all_lead_info_complete(session_id: str) -> bool:
    lead = get_lead_slots(session_id)
    required = ["name", "company"]
    has_contact = bool(lead.get("email") or lead.get("phone"))
    return all(lead.get(f) for f in required) and has_contact

def mark_closed(session_id: str) -> None:
    st = _SESSIONS.setdefault(session_id, {})
    st["is_closed"] = True; st["closed_at"] = _now_iso()

def is_session_closed(session_id: str) -> bool:
    return bool((_SESSIONS.get(session_id) or {}).get("is_closed"))

def cleanup_expired() -> None:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=SESSION_TTL_MINUTES)
    stale = []
    for sid, s in _SESSIONS.items():
        try: ts = datetime.fromisoformat(s.get("updated_at", _now_iso()))
        except Exception: ts = now
        if ts < cutoff: stale.append(sid)
    for sid in stale: _SESSIONS.pop(sid, None)