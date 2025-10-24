from __future__ import annotations
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

_SESSIONS: Dict[str, Dict[str, Any]] = {}

TURN_MEMORY_LIMIT = 12
SESSION_TTL_MINUTES = 60

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
        st["turns"] = turns[-TURN_MEMORY_LIMIT:]
    st["turns_count"] = st.get("turns_count", 0) + 1
    st["updated_at"] = _now_iso()

def recent_turns(session_id: str, n: int = 6) -> List[Dict[str, Any]]:
    return (_SESSIONS.get(session_id) or {}).get("turns", [])[-n:]

def get_turn_count(session_id: str) -> int:
    return int((_SESSIONS.get(session_id) or {}).get("turns_count", 0))

def update_summary(session_id: str) -> None:
    st = _SESSIONS.setdefault(session_id, {})
    last_user = next((t["content"] for t in reversed(st.get("turns", [])) if t["role"]=="user"), "")
    quick = (last_user[:150]+"…") if len(last_user) > 150 else (last_user or "-")
    st["summary"] = quick
    st["current_topic"] = st.get("current_topic") or "general"
    st["updated_at"] = _now_iso()

# ---- Lead slots ----
def get_lead_slots(session_id: str) -> Dict[str, Any]:
    st = _SESSIONS.setdefault(session_id, {})
    return st.setdefault("lead", {
        "name": None, "company": None, "email": None, "phone": None, "time": None
    })

def update_lead_slot(session_id: str, field: str, value: Optional[str]) -> None:
    if not value: return
    st = _SESSIONS.setdefault(session_id, {})
    lead = get_lead_slots(session_id)
    v = value.strip()
    if v and lead.get(field) != v:
        lead[field] = v
        st["last_slot_updated"] = field
        st["updated_at"] = _now_iso()

# CTA cooldown
def can_offer_cta(session_id: str, cooldown_turns: int = 2, max_attempts: int = 2) -> bool:
    st = _SESSIONS.setdefault(session_id, {})
    attempts = int(st.get("cta_attempts", 0))
    if attempts >= max_attempts:
        return False
    last = int(st.get("cta_last_turn", 0))
    return (get_turn_count(session_id) - last) >= cooldown_turns

def mark_cta_used(session_id: str) -> None:
    st = _SESSIONS.setdefault(session_id, {})
    st["cta_last_turn"] = get_turn_count(session_id)
    st["cta_attempts"] = int(st.get("cta_attempts", 0)) + 1
    st["updated_at"] = _now_iso()

def cleanup_expired() -> None:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=SESSION_TTL_MINUTES)
    for sid in list(_SESSIONS.keys()):
        try:
            ts = datetime.fromisoformat(_SESSIONS[sid].get("updated_at"))
        except Exception:
            ts = now
        if ts < cutoff:
            _SESSIONS.pop(sid, None)