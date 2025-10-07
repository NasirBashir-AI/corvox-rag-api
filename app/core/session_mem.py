# app/core/session_mem.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from app.core.config import DB_URL
from app.core.utils import pg_cursor

# We keep all per-session state in a single JSONB row.
# Shape we standardize on everywhere:
# {
#   "name": null, "phone": null, "email": null, "preferred_time": null,
#   "stage": null,               # lead flow stage: name|contact|time|notes|done
#   "lead_id": null,             # LEAD-... id if any
#   "lead_done": false,          # once true, don't restart unless user asks
#   "turns": [                   # rolling chat history (last 20 turns)
#     {"role": "user"|"assistant", "content": "text", "ts": "ISO8601"}
#   ]
# }

_DEF_STATE: Dict[str, Any] = {
    "name": None,
    "phone": None,
    "email": None,
    "preferred_time": None,
    "stage": None,
    "lead_id": None,
    "lead_done": False,
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

SQL_SELECT = """
SELECT state FROM corah_store.sessions WHERE session_id = %(sid)s;
"""

SQL_UPSERT = """
INSERT INTO corah_store.sessions (session_id, state, created_at, updated_at)
VALUES (%(sid)s, %(state)s::jsonb, now(), now())
ON CONFLICT (session_id)
DO UPDATE SET state = EXCLUDED.state, updated_at = now();
"""

SQL_DELETE = """
DELETE FROM corah_store.sessions WHERE session_id = %(sid)s;
"""


def _ensure_table() -> None:
    # Safe to run frequently; CREATE IF NOT EXISTS
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_CREATE)


def _merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        out[k] = v
    return out


def default_state() -> Dict[str, Any]:
    return json.loads(json.dumps(_DEF_STATE))  # deep copy


def get_state(session_id: str) -> Dict[str, Any]:
    _ensure_table()
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_SELECT, {"sid": session_id})
        row = cur.fetchone()
        if not row:
            return default_state()
        # psycopg/psycopg2 returns dict or str depending on driver; coerce
        st = row[0]
        if isinstance(st, dict):
            return _merge(default_state(), st)
        return _merge(default_state(), json.loads(st))


def set_state(session_id: str, **kwargs) -> Dict[str, Any]:
    st = get_state(session_id)
    # only set provided (non-None) keys; allow explicit None to clear via kwargs if desired
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