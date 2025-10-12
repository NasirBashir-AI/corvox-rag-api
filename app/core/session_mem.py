# app/core/session_mem.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

from app.core.config import (
    DB_URL,
)
from app.core.utils import pg_cursor

# -------------------------------------------------------------------
# State shape (JSONB) — one row per session_id
# -------------------------------------------------------------------
# {
#   "name": null, "phone": null, "email": null, "preferred_time": null, "timezone": null,
#   "lead_stage": null,        # name|contact|time|notes|done (controller-owned)
#   "lead_kind": null,         # e.g. "callback"
#   "lead_started_at": null,
#   "lead_done_at": null,
#   "lead_just_done": false,   # one-shot flag to close UI
#   "lead_nudge_at": null,
#   "lead_nudge_count": 0,
#   "asked_for_name_at": null,
#   "asked_for_contact_at": null,
#   "asked_for_time_at": null,
#   "asked_for_notes_at": null,
#   "current_topic": null,     # e.g., "WhatsApp chatbot for jewellery"
#   "intent": null,            # freeform
#   "objections": null,        # freeform
#   "next_step": null,         # freeform
#   "session_summary": "",     # rolling, compact summary (human-readable)
#   "turns": [ {"role":"user"|"assistant","content":"...","ts":"ISO8601Z"} ]  # last N turns
# }

_DEF_STATE: Dict[str, Any] = {
    "name": None,
    "phone": None,
    "email": None,
    "preferred_time": None,
    "timezone": None,

    "lead_stage": None,
    "lead_kind": None,
    "lead_started_at": None,
    "lead_done_at": None,
    "lead_just_done": False,

    "lead_nudge_at": None,
    "lead_nudge_count": 0,

    "asked_for_name_at": None,
    "asked_for_contact_at": None,
    "asked_for_time_at": None,
    "asked_for_notes_at": None,

    "current_topic": None,
    "intent": None,
    "objections": None,
    "next_step": None,

    "session_summary": "",
    "turns": [],
}

# Postgres DDL (schema/table-once)
SQL_CREATE = """
CREATE SCHEMA IF NOT EXISTS corah_store;

CREATE TABLE IF NOT EXISTS corah_store.sessions (
  session_id   TEXT PRIMARY KEY,
  state        JSONB NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sessions_updated_at
  ON corah_store.sessions (updated_at DESC);
"""

SQL_SELECT = "SELECT state FROM corah_store.sessions WHERE session_id = %(sid)s;"
SQL_UPSERT = """
INSERT INTO corah_store.sessions (session_id, state, created_at, updated_at)
VALUES (%(sid)s, %(state)s::jsonb, now(), now())
ON CONFLICT (session_id)
DO UPDATE SET state = EXCLUDED.state, updated_at = now();
"""
SQL_DELETE = "DELETE FROM corah_store.sessions WHERE session_id = %(sid)s;"

# -------------------------------------------------------------------
# Low-level helpers
# -------------------------------------------------------------------

def _ensure_table() -> None:
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_CREATE)

def _merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(patch or {})
    return out

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def default_state() -> Dict[str, Any]:
    # deep copy
    return json.loads(json.dumps(_DEF_STATE))

# -------------------------------------------------------------------
# Public: get/set state
# -------------------------------------------------------------------

def get_state(session_id: str) -> Dict[str, Any]:
    _ensure_table()
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_SELECT, {"sid": session_id})
        row = cur.fetchone()
        if not row:
            return default_state()
        val = row[0]
        if isinstance(val, dict):
            return _merge(default_state(), val)
        # driver might return str
        return _merge(default_state(), json.loads(val))

def set_state(session_id: str, **kwargs) -> Dict[str, Any]:
    st = get_state(session_id)
    for k, v in kwargs.items():
        st[k] = v
    _ensure_table()
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_UPSERT, {"sid": session_id, "state": json.dumps(st)})
    return st

def reset_session(session_id: str) -> None:
    _ensure_table()
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_DELETE, {"sid": session_id})

# -------------------------------------------------------------------
# Turns (short-window memory)
# -------------------------------------------------------------------

def append_turn(session_id: str, role: str, content: str, keep_last: int = 20) -> None:
    st = get_state(session_id)
    turns: List[Dict[str, Any]] = st.get("turns") or []
    turns.append({"role": role, "content": content, "ts": _now_iso()})
    if len(turns) > keep_last:
        turns = turns[-keep_last:]
    st["turns"] = turns
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_UPSERT, {"sid": session_id, "state": json.dumps(st)})

def recent_turns(session_id: str, n: int = 6) -> List[Dict[str, Any]]:
    st = get_state(session_id)
    turns: List[Dict[str, Any]] = st.get("turns") or []
    return turns[-n:]

# -------------------------------------------------------------------
# Asked-timestamps utils (cooldowns)
# -------------------------------------------------------------------

_FIELD_TO_ASKKEY = {
    "name": "asked_for_name_at",
    "contact": "asked_for_contact_at",
    "time": "asked_for_time_at",
    "notes": "asked_for_notes_at",
}

def mark_asked(session_id: str, field: str) -> None:
    key = _FIELD_TO_ASKKEY.get(field)
    if not key:
        return
    set_state(session_id, **{key: _now_iso()})

def recently_asked(session_id: str, field: str, cooldown_sec: int) -> bool:
    key = _FIELD_TO_ASKKEY.get(field)
    if not key:
        return False
    st = get_state(session_id)
    t_iso = st.get(key)
    if not t_iso:
        return False
    try:
        last = datetime.fromisoformat(t_iso)
        return (datetime.now(timezone.utc) - last).total_seconds() < max(0, int(cooldown_sec))
    except Exception:
        return False

def bump_nudge(session_id: str) -> None:
    st = get_state(session_id)
    cnt = int(st.get("lead_nudge_count") or 0) + 1
    set_state(session_id, lead_nudge_count=cnt, lead_nudge_at=_now_iso())

# -------------------------------------------------------------------
# Topic + Summary (rolling, light-weight)
# -------------------------------------------------------------------

def _detect_topic_from(text: str) -> Optional[str]:
    t = (text or "").lower()
    # very light heuristics — extend as needed
    if "whatsapp" in t:
        return "WhatsApp chatbot"
    if "chatbot" in t:
        return "Chatbot"
    if "pricing" in t or "cost" in t or "price" in t:
        return "Pricing"
    if "callback" in t or "call back" in t:
        return "Callback request"
    return None

def update_current_topic(session_id: str, last_user_text: str) -> None:
    topic = _detect_topic_from(last_user_text)
    if not topic:
        return
    st = get_state(session_id)
    prev = st.get("current_topic")
    if not prev or (topic.lower() not in prev.lower()):
        set_state(session_id, current_topic=topic)

def _fmt(val: Optional[str]) -> str:
    return val if val else "-"

def update_summary(session_id: str) -> None:
    """
    Build a compact rolling summary from structured slots.
    Keep it deterministic (no LLM here) so it's cheap and stable.
    """
    st = get_state(session_id)
    parts: List[str] = []

    # Core identity / contact
    parts.append(f"Name={_fmt(st.get('name'))}")
    parts.append(f"Phone={_fmt(st.get('phone'))}")
    parts.append(f"Email={_fmt(st.get('email'))}")
    parts.append(f"TimePref={_fmt(st.get('preferred_time'))}")
    if st.get("timezone"):
        parts.append(f"TZ={st['timezone']}")

    # Conversation intent/topic
    if st.get("current_topic"):
        parts.append(f"Topic={st['current_topic']}")
    if st.get("intent"):
        parts.append(f"Intent={st['intent']}")
    if st.get("objections"):
        parts.append(f"Objections={st['objections']}")
    if st.get("next_step"):
        parts.append(f"Next={st['next_step']}")

    # Lead stage
    if st.get("lead_stage"):
        parts.append(f"LeadStage={st['lead_stage']}")

    summary = " | ".join(parts)
    set_state(session_id, session_summary=summary)

# -------------------------------------------------------------------
# Transcript search (for quick “as I said earlier” lookups)
# -------------------------------------------------------------------

def search_transcript(session_id: str, keywords: List[str], max_chars: int = 400) -> Optional[str]:
    """
    Tiny keyword scan over the stored turns; returns one compact snippet.
    This is deliberately simple (fast, deterministic).
    """
    st = get_state(session_id)
    turns: List[Dict[str, Any]] = st.get("turns") or []
    if not turns or not keywords:
        return None

    keys = [k.lower() for k in keywords if k]
    # scan from newest backward to find the most recent matching user turn
    for t in reversed(turns):
        if t.get("role") != "user":
            continue
        txt = (t.get("content") or "")
        low = txt.lower()
        if all(k in low for k in keys):
            snip = txt.strip()
            if len(snip) > max_chars:
                snip = snip[: max_chars - 3] + "..."
            ts = t.get("ts") or ""
            return f"{snip}  (said earlier at {ts})"
    return None