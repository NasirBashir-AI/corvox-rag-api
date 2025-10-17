# app/core/session_mem.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from app.core.config import DB_URL, CTA_COOLDOWN_TURNS, CTA_MAX_ATTEMPTS
from app.core.utils import pg_cursor

"""
Session memory is a single JSONB blob per session_id.
We intentionally keep this schema flexible and avoid DB migrations by adding
new keys inside `state`.

State shape (illustrative; keys may be absent if never set):
{
  "name": null, "phone": null, "email": null, "preferred_time": null,
  "company": null,
  "session_summary": "",          # short rolling summary
  "current_topic": null,          # e.g., "WhatsApp chatbot for jewellery"
  "lead_stage": null,             # name|contact|time|notes|done
  "lead_started_at": null, "lead_done_at": null,
  "lead_nudge_at": null, "lead_nudge_count": 0,

  # ask cooldown stamps
  "asked_for_name_at": null, "asked_for_contact_at": null,
  "asked_for_email_at": null, "asked_for_phone_at": null,
  "asked_for_time_at": null, "asked_for_notes_at": null,

  # turns + counters (for CTA cooldowns and behaviour guards)
  "turns": [ {role, content, ts} ],
  "turns_count": 0,               # incremented on every append_turn
  "cta_attempts": 0,              # total CTA offers made
  "cta_last_turn": 0,             # turn index at last CTA
  "last_activity_at": null,       # ISO timestamp of last user/assistant turn

  # lifecycle
  "session_closed": false         # once true, /chat should respond with end_session
}
"""

_DEF_STATE: Dict[str, Any] = {
    "name": None,
    "phone": None,
    "email": None,
    "preferred_time": None,
    "company": None,
    "session_summary": "",
    "current_topic": None,
    "lead_stage": None,
    "lead_started_at": None,
    "lead_done_at": None,
    "lead_nudge_at": None,
    "lead_nudge_count": 0,
    "asked_for_name_at": None,
    "asked_for_contact_at": None,
    "asked_for_email_at": None,
    "asked_for_phone_at": None,
    "asked_for_time_at": None,
    "asked_for_notes_at": None,
    "turns": [],
    "turns_count": 0,
    "cta_attempts": 0,
    "cta_last_turn": 0,
    "last_activity_at": None,
    "session_closed": False,
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


# ---------- low-level utils ----------

def _ensure_table() -> None:
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_CREATE)

def _merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(patch or {})
    return out

def default_state() -> Dict[str, Any]:
    # deep copy
    return json.loads(json.dumps(_DEF_STATE))


# ---------- CRUD ----------

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
    # always refresh updated_at + last_activity when state mutates
    st["last_activity_at"] = datetime.now(timezone.utc).isoformat()
    _ensure_table()
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_UPSERT, {"sid": session_id, "state": json.dumps(st)})
    return st

def reset_session(session_id: str) -> None:
    _ensure_table()
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_DELETE, {"sid": session_id})


# ---------- turns ----------

def append_turn(session_id: str, role: str, content: str, keep_last: int = 30) -> None:
    st = get_state(session_id)
    turns: List[Dict[str, Any]] = st.get("turns") or []
    turns.append({"role": role, "content": content, "ts": datetime.utcnow().isoformat() + "Z"})
    if len(turns) > keep_last:
        turns = turns[-keep_last:]
    # counters
    turns_count = int(st.get("turns_count") or 0) + 1
    st["turns"] = turns
    st["turns_count"] = turns_count
    st["last_activity_at"] = datetime.now(timezone.utc).isoformat()
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_UPSERT, {"sid": session_id, "state": json.dumps(st)})

def recent_turns(session_id: str, n: int = 6) -> List[Dict[str, Any]]:
    st = get_state(session_id)
    turns: List[Dict[str, Any]] = st.get("turns") or []
    return turns[-n:]

def get_turns_count(session_id: str) -> int:
    st = get_state(session_id)
    return int(st.get("turns_count") or 0)


# ---------- CTA helpers (cooldown + attempts) ----------

def can_offer_cta(session_id: str) -> bool:
    """
    True if we are allowed to make another CTA (e.g., 'Shall I arrange a discovery call?').
    Enforces:
      - at least CTA_COOLDOWN_TURNS turns since last CTA
      - CTA_MAX_ATTEMPTS total cap
    """
    st = get_state(session_id)
    attempts = int(st.get("cta_attempts") or 0)
    last_turn = int(st.get("cta_last_turn") or 0)
    turn_idx  = int(st.get("turns_count") or 0)
    if attempts >= CTA_MAX_ATTEMPTS:
        return False
    if (turn_idx - last_turn) < CTA_COOLDOWN_TURNS:
        return False
    return True

def note_cta_attempt(session_id: str) -> None:
    st = get_state(session_id)
    attempts = int(st.get("cta_attempts") or 0) + 1
    turn_idx  = int(st.get("turns_count") or 0)
    set_state(session_id, cta_attempts=attempts, cta_last_turn=turn_idx)


# ---------- ask tracking / cooldown helpers ----------

def mark_asked(session_id: str, field: str) -> None:
    """
    field ∈ {name, contact, email, phone, time, notes}
    Stamps asked_for_<field>_at with now().
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    key = f"asked_for_{field}_at"
    set_state(session_id, **{key: now_iso})

def recently_asked(session_id: str, field: str, cooldown_sec: int) -> bool:
    st = get_state(session_id)
    key = f"asked_for_{field}_at"
    iso = st.get(key)
    if not iso:
        return False
    try:
        dt = datetime.fromisoformat(iso)
    except Exception:
        return False
    return (datetime.now(timezone.utc) - dt).total_seconds() < cooldown_sec


# ---------- rolling summary (two-layer memory) ----------

def _short(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    s = " ".join((v or "").split())
    return s[:140] if len(s) > 140 else s

def update_summary(session_id: str) -> str:
    """
    Build a compact, human-usable summary string from current state.
    We keep it short so the LLM can reliably use it to stay on topic.
    """
    st = get_state(session_id)
    name = st.get("name")
    company = st.get("company")
    topic = st.get("current_topic")
    phone = st.get("phone")
    email = st.get("email")
    when = st.get("preferred_time")

    parts: List[str] = []
    if company:
        parts.append(f"company: {company}")
    if topic:
        parts.append(f"topic: {topic}")
    if when:
        parts.append(f"time_pref: {when}")
    if name:
        parts.append(f"name: {name}")
    if phone:
        parts.append(f"phone: {phone}")
    if email:
        parts.append(f"email: {email}")

    summary = "; ".join(parts)
    summary = _short(summary) or ""
    set_state(session_id, session_summary=summary)
    return summary


# ---------- lifecycle ----------

def set_session_closed(session_id: str, closed: bool = True) -> None:
    set_state(session_id, session_closed=closed)

def is_session_closed(session_id: str) -> bool:
    st = get_state(session_id)
    return bool(st.get("session_closed"))

# ---------- optional: keep a compact lead recap in session state ----------
from typing import Any, Dict  # (already present at top; keep if needed)

def set_lead_backup(session_id: str, report: Dict[str, Any]) -> None:
    """
    Store a compact lead recap/report in the per-session state so the frontend
    or admin tools can read it even if DB write fails. Safe, JSON-serializable.
    """
    try:
        # Minimal: just persist under 'lead_backup'
        set_state(session_id, lead_backup=report)
    except Exception:
        # Never crash the API because of backup
        pass