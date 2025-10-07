# app/core/session_mem.py
from __future__ import annotations
from typing import Any, Dict, List

# In-memory, per-session scratchpad.
# This is intentionally simple (no expiry, no persistence).
# Fields we commonly store:
#   state["name"], state["phone"], state["email"], state["preferred_time"]
#   state["turns"] = [{"role":"user"|"assistant", "content":"..."} ...]
#   state["lead_stage"] = one of: None|"name"|"contact"|"time"|"notes"|"done"

_STORE: Dict[str, Dict[str, Any]] = {}

def get_state(session_id: str) -> Dict[str, Any]:
    """Get (and lazily create) the state dict for a session."""
    if session_id not in _STORE:
        _STORE[session_id] = {"turns": [], "lead_stage": None}
    return _STORE[session_id]

def set_state(session_id: str, **kwargs: Any) -> None:
    """Merge arbitrary keys into the session state."""
    st = get_state(session_id)
    st.update({k: v for k, v in kwargs.items() if v is not None})

def clear_session(session_id: str) -> None:
    """Drop all memory for a session (use when you explicitly end a chat)."""
    _STORE.pop(session_id, None)

# ---- Turns memory (for classifier context) ----

def append_turn(session_id: str, role: str, content: str, max_keep: int = 8) -> None:
    """
    Append a conversation turn and keep only the last `max_keep`.
    role: "user" or "assistant"
    """
    st = get_state(session_id)
    turns: List[Dict[str, str]] = st.setdefault("turns", [])
    turns.append({"role": role, "content": content or ""})
    if len(turns) > max_keep:
        # keep the most recent `max_keep`
        st["turns"] = turns[-max_keep:]

def recent_turns(session_id: str, n: int = 4) -> List[Dict[str, str]]:
    """Return the last n turns (role/content) for classification."""
    st = get_state(session_id)
    turns: List[Dict[str, str]] = st.get("turns", [])
    if n <= 0:
        return []
    return turns[-n:]

# ---- Optional helpers for lead flow bookkeeping (if you want to use them) ----

def set_lead_stage(session_id: str, stage: str | None) -> None:
    st = get_state(session_id)
    st["lead_stage"] = stage

def get_lead_stage(session_id: str) -> str | None:
    st = get_state(session_id)
    return st.get("lead_stage")

def lead_in_progress(session_id: str) -> bool:
    return bool(get_lead_stage(session_id))