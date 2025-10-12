# app/core/session_mem.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta

from app.core.config import DB_URL
from app.core.utils import pg_cursor

# Per-session durable JSONB state.
_DEF_STATE: Dict[str, Any] = {
    "name": None,
    "phone": None,
    "email": None,
    "preferred_time": None,

    "lead_stage": None,          # name|contact|time|notes|done
    "lead_id": None,
    "lead_done_at": None,
    "lead_started_at": None,
    "lead_just_done": False,
    "lead_nudge_at": None,
    "lead_nudge_count": 0,

    # asked-timestamps for cooldowns
    "asked_for_name_at": None,
    "asked_for_contact_at": None,
    "asked_for_time_at": None,
    "asked_for_notes_at": None,

    # light memory
    "session_summary": "",
    "current_topic": "",
    "pending_offer": None,

    "turns": [],
}

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


def _ensure_table() -> None:
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_CREATE)


def _merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(patch or {})
    return out


def default_state() -> Dict[str, Any]:
    return json.loads(json.dumps(_DEF_STATE))


def get_state(session_id: str) -> Dict[str, Any]:
    _ensure_table()
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_SELECT, {"sid": session_id})
        row = cur.fetchone()
        if not row:
            return default_state()
        st = row[0]
        if isinstance(st, dict):
            return _merge(default_state(), st)
        return _merge(default_state(), json.loads(st))


def set_state(session_id: str, **kwargs) -> Dict[str, Any]:
    st = get_state(session_id)
    for k, v in kwargs.items():
        st[k] = v
    _ensure_table()
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_UPSERT, {"sid": session_id, "state": json.dumps(st)})
    return st


def append_turn(session_id: str, role: str, content: str, keep_last: int = 20) -> None:
    st = get_state(session_id)
    turns: List[Dict[str, Any]] = st.get("turns") or []
    turns.append({"role": role, "content": content, "ts": datetime.utcnow().isoformat() + "Z"})
    if len(turns) > keep_last:
        turns = turns[-keep_last:]
    st["turns"] = turns
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_UPSERT, {"sid": session_id, "state": json.dumps(st)})


def recent_turns(session_id: str, n: int = 6) -> List[Dict[str, Any]]:
    st = get_state(session_id)
    turns: List[Dict[str, Any]] = st.get("turns") or []
    return turns[-n:]


# ---------- Pending offer / topic / summary ----------

def get_pending_offer(session_id: str) -> Optional[str]:
    return (get_state(session_id) or {}).get("pending_offer")

def set_pending_offer(session_id: str, text: str) -> None:
    set_state(session_id, pending_offer=(text or "").strip() or None)

def clear_pending_offer(session_id: str) -> None:
    set_state(session_id, pending_offer=None)

def set_current_topic(session_id: str, topic: Optional[str]) -> None:
    topic = (topic or "").strip()
    if topic:
        set_state(session_id, current_topic=topic)

def update_summary(session_id: str, *, intent: Optional[str] = None) -> str:
    st = get_state(session_id) or {}
    topic = (st.get("current_topic") or "").strip()
    name = st.get("name") or "-"
    phone = st.get("phone") or "-"
    email = st.get("email") or "-"
    ptime = st.get("preferred_time") or "-"
    intent = (intent or "").strip()

    parts = []
    if topic:
        parts.append(f"Topic: {topic}")
    parts.append(f"Name: {name}")
    parts.append(f"Phone: {phone}")
    parts.append(f"Email: {email}")
    parts.append(f"Preferred time: {ptime}")
    if intent:
        parts.append(f"Intent: {intent}")

    summary = " | ".join(parts)
    if len(summary) > 400:
        summary = summary[:397] + "..."
    set_state(session_id, session_summary=summary)
    return summary


# ---------- Compatibility helpers expected by capture.py ----------

_ASK_FIELD_MAP = {
    "name": "asked_for_name_at",
    "contact": "asked_for_contact_at",   # phone or email
    "time": "asked_for_time_at",
    "notes": "asked_for_notes_at",
}

def _parse_iso_maybe_z(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    s = ts.strip()
    try:
        # support both "...Z" and plain ISO
        if s.endswith("Z"):
            s = s[:-1]
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except Exception:
        return None

def mark_asked(session_id: str, field: str, ts_iso: Optional[str] = None) -> None:
    """
    Stamp asked_for_<field>_at with now (UTC) or provided ISO timestamp.
    Fields: name | contact | time | notes
    """
    key = _ASK_FIELD_MAP.get(field)
    if not key:
        return
    ts = (ts_iso or (datetime.utcnow().isoformat() + "Z"))
    set_state(session_id, **{key: ts})

def recently_asked(session_id: str, field: str, cooldown_sec: int) -> bool:
    """
    True if asked_for_<field>_at is within cooldown_sec.
    """
    key = _ASK_FIELD_MAP.get(field)
    if not key:
        return False
    st = get_state(session_id)
    ts = _parse_iso_maybe_z(st.get(key))
    if not ts:
        return False
    now = datetime.now(timezone.utc)
    return (now - ts) < timedelta(seconds=max(0, int(cooldown_sec or 0)))

def bump_nudge(session_id: str, now_iso: Optional[str] = None) -> None:
    """
    Increment lead_nudge_count and set lead_nudge_at.
    """
    st = get_state(session_id)
    count = int(st.get("lead_nudge_count") or 0) + 1
    set_state(session_id, lead_nudge_count=count, lead_nudge_at=(now_iso or datetime.utcnow().isoformat() + "Z"))


def reset_session(session_id: str) -> None:
    _ensure_table()
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_DELETE, {"sid": session_id})