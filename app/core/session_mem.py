# app/core/session_mem.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from app.core.config import DB_URL
from app.core.utils import pg_cursor

# Per-session state in one JSONB row.
# {
#   "name": null, "phone": null, "email": null, "preferred_time": null,
#   "company": null,
#   "session_summary": "",          # rolling short summary we keep updated
#   "current_topic": null,          # e.g., "WhatsApp chatbot for jewellery"
#   "lead_stage": null,             # name|contact|time|notes|done
#   "lead_started_at": null, "lead_done_at": null,
#   "lead_nudge_at": null, "lead_nudge_count": 0,
#   "asked_for_name_at": null, "asked_for_contact_at": null,
#   "asked_for_email_at": null, "asked_for_phone_at": null,
#   "asked_for_time_at": null, "asked_for_notes_at": null,
#   "turns": [ {role, content, ts} ],
#   # ---- Phase 2 additions ----
#   "sentiment": null,              # positive | neutral | frustrated
#   "intent_level": null,           # cold | warm | hot
#   "last_asked": null,             # name|company|email|phone|time|notes
#   "lead_backup": null,            # snapshot dict of recap before save
#   "notes": null,                  # short internal summary (incl. sentiment/intent)
#   "priority": null,               # hot|warm|cold (optional)
#   "end_session": false,           # signal UI to disable input
#   "last_user_at": null,           # ISO ts of most recent user message
#   "inactivity_warn_at": null,     # ISO ts when warning was issued
#   "inactivity_close_at": null     # ISO ts when auto-close happened
# }

# ...imports + existing code stay the same...

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

    # --- NEW: counters/cta/inactivity ---
    "turns_count": 0,
    "cta_attempts": 0,
    "cta_last_turn": -1,
    "last_user_at": None,
    "inactivity_warned": False,

    # optional diagnostics
    "sentiment": None,
    "intent_level": None,
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_state() -> Dict[str, Any]:
    # deep copy
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


# ...keep SQL_* and helpers unchanged...

def append_turn(session_id: str, role: str, content: str, keep_last: int = 30) -> None:
    st = get_state(session_id)
    turns: List[Dict[str, Any]] = st.get("turns") or []
    turns.append({"role": role, "content": content, "ts": datetime.utcnow().isoformat() + "Z"})
    if len(turns) > keep_last:
        turns = turns[-keep_last:]
    # NEW: increment turns_count each time we add a turn
    turns_count = int(st.get("turns_count") or 0) + 1
    st["turns"] = turns
    st["turns_count"] = turns_count
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_UPSERT, {"sid": session_id, "state": json.dumps(st)})

# Optional tiny helpers (used by cooldown logic; safe to keep even if unused)
def get_turns_count(session_id: str) -> int:
    return int(get_state(session_id).get("turns_count") or 0)

def bump_cta(session_id: str, turn_idx: int) -> None:
    st = get_state(session_id)
    set_state(session_id,
        cta_attempts=int(st.get("cta_attempts") or 0) + 1,
        cta_last_turn=int(turn_idx),
    )

def recent_turns(session_id: str, n: int = 6) -> List[Dict[str, Any]]:
    st = get_state(session_id)
    turns: List[Dict[str, Any]] = st.get("turns") or []
    return turns[-n:]


def reset_session(session_id: str) -> None:
    _ensure_table()
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_DELETE, {"sid": session_id})


# ---------- ask tracking / cooldown helpers ----------

def mark_asked(session_id: str, field: str) -> None:
    """
    field ∈ {name, contact, email, phone, time, notes, company}
    Stamps asked_for_<field>_at with now() and sets last_asked.
    """
    now_iso = _now_iso()
    key = f"asked_for_{field}_at"
    st = get_state(session_id)
    st[key] = now_iso
    st["last_asked"] = field
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_UPSERT, {"sid": session_id, "state": json.dumps(st)})


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


# ---------- Phase 2 helpers (slots, signals, recap/backup, closure) ----------

SLOT_KEYS = ("name", "company", "email", "phone", "preferred_time")

def get_slots(session_id: str) -> Dict[str, Optional[str]]:
    st = get_state(session_id)
    return {k: st.get(k) for k in SLOT_KEYS}

def set_slots(session_id: str, **slots) -> Dict[str, Any]:
    """
    Safe slot update (e.g., name/company/email/phone/preferred_time).
    """
    allowed = {k: v for k, v in slots.items() if k in SLOT_KEYS}
    return set_state(session_id, **allowed)

def apply_corrections(session_id: str, **slots) -> Dict[str, Any]:
    """
    Accept user corrections (e.g., name/email edits) and persist immediately.
    """
    st = set_slots(session_id, **slots)
    update_summary(session_id)
    return st

def set_signals(session_id: str, sentiment: Optional[str] = None, intent_level: Optional[str] = None,
                priority: Optional[str] = None) -> Dict[str, Any]:
    st = get_state(session_id)
    if sentiment is not None:
        st["sentiment"] = sentiment
    if intent_level is not None:
        st["intent_level"] = intent_level
    if priority is not None:
        st["priority"] = priority
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_UPSERT, {"sid": session_id, "state": json.dumps(st)})
    return st

def set_current_topic(session_id: str, topic: Optional[str]) -> Dict[str, Any]:
    return set_state(session_id, current_topic=topic)

def set_lead_backup(session_id: str, backup: Any) -> Dict[str, Any]:
    return set_state(session_id, lead_backup=backup)

def set_notes(session_id: str, notes: Optional[str]) -> Dict[str, Any]:
    return set_state(session_id, notes=notes)

def mark_lead_started(session_id: str) -> Dict[str, Any]:
    return set_state(session_id, lead_stage="started", lead_started_at=_now_iso())

def mark_lead_done(session_id: str) -> Dict[str, Any]:
    return set_state(session_id, lead_stage="done", lead_done_at=_now_iso())

def mark_end_session(session_id: str, value: bool = True) -> Dict[str, Any]:
    return set_state(session_id, end_session=value)

def record_user_activity(session_id: str) -> Dict[str, Any]:
    """
    Update last_user_at to now (call on each user turn if you want strict inactivity tracking).
    """
    return set_state(session_id, last_user_at=_now_iso())

def mark_inactivity_warn(session_id: str) -> Dict[str, Any]:
    return set_state(session_id, inactivity_warn_at=_now_iso())

def mark_inactivity_close(session_id: str) -> Dict[str, Any]:
    return set_state(session_id, inactivity_close_at=_now_iso(), end_session=True)