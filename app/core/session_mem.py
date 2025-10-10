# app/core/session_mem.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta

from app.core.config import DB_URL
from app.core.utils import pg_cursor

"""
Per-session durable state stored as a single JSONB row.

Standardized shape (keys may be None):
{
  "name": null,
  "phone": null,
  "email": null,
  "preferred_time": null,

  # Lead flow controller state
  "lead_stage": null,              # 'name' | 'contact' | 'time' | 'notes' | 'done'
  "lead_kind": null,               # e.g., 'callback'
  "lead_started_at": null,         # ISO ts once nudged/started
  "lead_done_at": null,            # ISO ts when completed
  "lead_just_done": false,         # one-shot flag to allow UI to close once

  # Nudge / anti-pushy guards
  "lead_nudge_at": null,           # last time we nudged
  "lead_nudge_count": 0,           # budgeted per session

  # Ask-cooldowns (one-ask-per-turn support)
  "asked_for_name_at": null,
  "asked_for_contact_at": null,    # phone/email combined
  "asked_for_time_at": null,
  "asked_for_notes_at": null,

  # Rolling chat history
  "turns": [
    {"role": "user"|"assistant", "content": "text", "ts": "ISO8601"}
  ]
}
"""

# ---------- defaults & SQL ----------

_DEF_STATE: Dict[str, Any] = {
    "name": None,
    "phone": None,
    "email": None,
    "preferred_time": None,

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


# ---------- helpers ----------

def _ensure_table() -> None:
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_CREATE)

def _merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(patch or {})
    return out

def default_state() -> Dict[str, Any]:
    # deep copy of default
    return json.loads(json.dumps(_DEF_STATE))

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _from_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        # legacy Z-suffixed fallback
        try:
            if ts.endswith("Z"):
                ts2 = ts[:-1] + "+00:00"
                return datetime.fromisoformat(ts2)
        except Exception:
            return None
    return None


# ---------- core API ----------

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
    turns.append({"role": role, "content": content, "ts": _iso_now()})
    if len(turns) > keep_last:
        turns = turns[-keep_last:]
    st["turns"] = turns
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_UPSERT, {"sid": session_id, "state": json.dumps(st)})

def recent_turns(session_id: str, n: int = 4) -> List[Dict[str, Any]]:
    st = get_state(session_id)
    turns: List[Dict[str, Any]] = st.get("turns") or []
    return turns[-n:]

def reset_session(session_id: str) -> None:
    _ensure_table()
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_DELETE, {"sid": session_id})


# ---------- ask/nudge helpers (used by controller & orchestrator) ----------

# Mark that we asked for a particular field (name/contact/time/notes)
def mark_asked(session_id: str, field: str) -> None:
    key = None
    f = (field or "").strip().lower()
    if f == "name":
        key = "asked_for_name_at"
    elif f in ("contact", "phone", "email", "phone_or_email"):
        key = "asked_for_contact_at"
    elif f == "time":
        key = "asked_for_time_at"
    elif f == "notes":
        key = "asked_for_notes_at"
    if key:
        set_state(session_id, **{key: _iso_now()})

# Has this field been asked very recently?
def recently_asked(session_id: str, field: str, cooldown_sec: int) -> bool:
    st = get_state(session_id)
    key = None
    f = (field or "").strip().lower()
    if f == "name":
        key = "asked_for_name_at"
    elif f in ("contact", "phone", "email", "phone_or_email"):
        key = "asked_for_contact_at"
    elif f == "time":
        key = "asked_for_time_at"
    elif f == "notes":
        key = "asked_for_notes_at"
    if not key:
        return False
    ts = _from_iso(st.get(key))
    if not ts:
        return False
    return (datetime.now(timezone.utc) - ts) < timedelta(seconds=cooldown_sec)

# Increment nudge budget and stamp time
def bump_nudge(session_id: str, when_iso: Optional[str] = None) -> None:
    st = get_state(session_id)
    count = int(st.get("lead_nudge_count") or 0) + 1
    set_state(session_id, lead_nudge_count=count, lead_nudge_at=(when_iso or _iso_now()))

# Convenience: find the most recent asked_* field
def last_asked_field(session_id: str) -> Optional[str]:
    st = get_state(session_id)
    pairs = []
    for fld, key in (
        ("name", "asked_for_name_at"),
        ("contact", "asked_for_contact_at"),
        ("time", "asked_for_time_at"),
        ("notes", "asked_for_notes_at"),
    ):
        ts = _from_iso(st.get(key))
        if ts:
            pairs.append((fld, ts))
    if not pairs:
        return None
    pairs.sort(key=lambda kv: kv[1])
    return pairs[-1][0]