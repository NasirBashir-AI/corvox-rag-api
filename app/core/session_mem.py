# app/core/session_mem.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta

from app.core.config import DB_URL
from app.core.utils import pg_cursor

# We keep all per-session state in a single JSONB row.
# Canonical shape:
# {
#   "name": null, "phone": null, "email": null, "preferred_time": null,
#   "lead_stage": null,          # name|contact|time|notes|done
#   "lead_kind": null,           # e.g. "callback"
#   "lead_id": null,
#   "lead_done": false,          # legacy boolean
#   "lead_started_at": null,     # ISO
#   "lead_done_at": null,        # ISO
#   "lead_nudge_at": null,       # ISO (last time we nudged to start)
#   "lead_nudge_count": 0,
#   "asked_for_name_at": null,   # ISO (per-field cool-down stamps)
#   "asked_for_phone_at": null,
#   "asked_for_email_at": null,
#   "asked_for_time_at": null,
#   "asked_for_notes_at": null,
#   "turns": [ {role, content, ts} ]   # rolling chat history
# }

_DEF_STATE: Dict[str, Any] = {
    "name": None,
    "phone": None,
    "email": None,
    "preferred_time": None,

    # Lead flow
    "lead_stage": None,
    "lead_kind": None,
    "lead_id": None,
    "lead_done": False,      # legacy flag
    "lead_started_at": None,
    "lead_done_at": None,

    # Nudger / guard
    "lead_nudge_at": None,
    "lead_nudge_count": 0,

    # Per-field cool-downs
    "asked_for_name_at": None,
    "asked_for_phone_at": None,
    "asked_for_email_at": None,
    "asked_for_time_at": None,
    "asked_for_notes_at": None,

    # Chat history
    "turns": [],
    # ---- Legacy/compat ----
    "stage": None,  # old key; we still accept it on read and map into lead_stage
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


# ---------- internals ----------

def _ensure_table() -> None:
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_CREATE)

def _merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(patch or {})
    return out

def _deepcopy_state(st: Dict[str, Any]) -> Dict[str, Any]:
    # cheap deep copy
    return json.loads(json.dumps(st))

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------- public api ----------

def default_state() -> Dict[str, Any]:
    return _deepcopy_state(_DEF_STATE)

def get_state(session_id: str) -> Dict[str, Any]:
    _ensure_table()
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_SELECT, {"sid": session_id})
        row = cur.fetchone()
        if not row:
            return default_state()
        raw = row[0]
        st: Dict[str, Any]
        if isinstance(raw, dict):
            st = raw
        else:
            st = json.loads(raw)

    # Merge onto defaults and normalize legacy keys
    st = _merge(default_state(), st)

    # Compat: map legacy "stage" -> "lead_stage" if present
    if st.get("lead_stage") is None and st.get("stage"):
        st["lead_stage"] = st.get("stage")

    return st

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

def recent_turns(session_id: str, n: int = 4) -> List[Dict[str, Any]]:
    st = get_state(session_id)
    turns: List[Dict[str, Any]] = st.get("turns") or []
    return turns[-n:]

def reset_session(session_id: str) -> None:
    _ensure_table()
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_DELETE, {"sid": session_id})


# ---------- guard helpers (cool-downs & nudges) ----------

def mark_asked(session_id: str, field: str) -> None:
    """
    Stamp asked_for_<field>_at with current time (UTC ISO).
    field in {"name","phone","email","time","notes"}.
    """
    key = f"asked_for_{field}_at"
    set_state(session_id, **{key: _now_iso()})

def recently_asked(session_id: str, field: str, cooldown_sec: int) -> bool:
    """
    Return True if asked_for_<field>_at is within cooldown_sec.
    """
    st = get_state(session_id)
    key = f"asked_for_{field}_at"
    iso = st.get(key)
    if not iso:
        return False
    try:
        then = datetime.fromisoformat(iso)
        now = datetime.now(timezone.utc)
        return (now - then) < timedelta(seconds=cooldown_sec)
    except Exception:
        return False

def bump_nudge(session_id: str, now_iso: Optional[str] = None) -> int:
    """
    Increment lead_nudge_count and set/update lead_nudge_at.
    Returns the updated count.
    """
    st = get_state(session_id)
    cnt = int(st.get("lead_nudge_count") or 0) + 1
    set_state(session_id, lead_nudge_count=cnt, lead_nudge_at=(now_iso or _now_iso()))
    return cnt