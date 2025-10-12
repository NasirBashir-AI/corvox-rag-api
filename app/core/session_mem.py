# app/core/session_mem.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

from app.core.config import DB_URL
from app.core.utils import pg_cursor

"""
Single row of per-session state stored in JSONB.

Fields we rely on across the stack:
- name, phone, email, preferred_time, timezone
- current_topic: short phrase summarising what we're talking about now
- session_summary: rolling summary string we keep concise and up to date
- lead_stage: 'name' | 'contact' | 'time' | 'notes' | 'done' | None
- lead_started_at, lead_done_at
- lead_nudge_count, lead_nudge_at
- asked_for_name_at, asked_for_contact_at, asked_for_time_at, asked_for_notes_at
- turns: [{role:'user'|'assistant', content:str, ts:ISO}]

Helpers:
- mark_asked(field): stamps asked_for_<field>_at
- recently_asked(field, cooldown_sec): bool
- bump_nudge(now_iso)
- update_summary(session_id, last_user_text=None): maintain session_summary and current_topic
- transcript_search(session_id, needle, max_chars=280): find one snippet from older turns
"""

_DEF_STATE: Dict[str, Any] = {
    "name": None,
    "phone": None,
    "email": None,
    "preferred_time": None,
    "timezone": None,
    "current_topic": None,
    "session_summary": "",

    "lead_stage": None,
    "lead_started_at": None,
    "lead_done_at": None,
    "lead_nudge_count": 0,
    "lead_nudge_at": None,

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


def _ensure_table() -> None:
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_CREATE)


def _merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(patch or {})
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
        st = row[0]
        if isinstance(st, dict):
            return _merge(default_state(), st)
        return _merge(default_state(), json.loads(st))


def set_state(session_id: str, **kwargs) -> Dict[str, Any]:
    st = get_state(session_id)
    for k, v in kwargs.items():
        st[k] = v
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_UPSERT, {"sid": session_id, "state": json.dumps(st)})
    return st


def append_turn(session_id: str, role: str, content: str, keep_last: int = 24) -> None:
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


def reset_session(session_id: str) -> None:
    _ensure_table()
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_DELETE, {"sid": session_id})


# -------- cool-down helpers (aligned keys) --------

def _field_key(field: str) -> str:
    # normalized: 'name'|'contact'|'time'|'notes'
    return {
        "name": "asked_for_name_at",
        "contact": "asked_for_contact_at",
        "time": "asked_for_time_at",
        "notes": "asked_for_notes_at",
    }.get(field, f"asked_for_{field}_at")


def mark_asked(session_id: str, field: str) -> None:
    key = _field_key(field)
    now_iso = datetime.now(timezone.utc).isoformat()
    set_state(session_id, **{key: now_iso})


def recently_asked(session_id: str, field: str, cooldown_sec: int) -> bool:
    key = _field_key(field)
    st = get_state(session_id)
    t_iso = st.get(key)
    if not t_iso:
        return False
    try:
        dt = datetime.fromisoformat(t_iso)
        return (datetime.now(timezone.utc) - dt).total_seconds() < cooldown_sec
    except Exception:
        return False


def bump_nudge(session_id: str, now_iso: Optional[str] = None) -> None:
    st = get_state(session_id)
    cnt = int(st.get("lead_nudge_count") or 0) + 1
    now_iso = now_iso or datetime.now(timezone.utc).isoformat()
    set_state(session_id, lead_nudge_count=cnt, lead_nudge_at=now_iso)


# -------- summary + topic maintenance --------

def _shorten(text: str, limit: int = 220) -> str:
    t = (text or "").strip()
    return t if len(t) <= limit else t[:limit - 1] + "…"


def update_summary(session_id: str, last_user_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Keep a compact rolling summary + current_topic.
    Very lightweight heuristic: prefer explicit domain nouns from the last user text,
    otherwise keep the previous topic.
    """
    st = get_state(session_id)
    name = st.get("name")
    phone = st.get("phone")
    email = st.get("email")
    pref_time = st.get("preferred_time")
    topic = st.get("current_topic")

    # Try to infer/update topic from latest user text if it contains a concrete noun
    inferred = None
    t = (last_user_text or "").lower()
    for kw in ("whatsapp", "chatbot", "reconciliation", "inventory", "student recruitment",
               "accountancy", "voice agent", "ai agent", "lead capture", "pricing", "callback"):
        if kw in t:
            inferred = kw
            break
    if inferred:
        topic = inferred

    # Build a compact one-liner
    bits: List[str] = []
    if topic:
        bits.append(f"Topic: {topic}")
    if name:
        bits.append(f"Name: {name}")
    if phone:
        bits.append(f"Phone: {phone}")
    if email:
        bits.append(f"Email: {email}")
    if pref_time:
        bits.append(f"Preferred time: {pref_time}")

    summary = " | ".join(bits) if bits else (st.get("session_summary") or "")
    summary = _shorten(summary, 360)

    return set_state(session_id, session_summary=summary, current_topic=topic)


def transcript_search(session_id: str, needle: str, max_chars: int = 280) -> Optional[str]:
    """Very small keyword scan over all stored turns; return first short hit."""
    if not needle:
        return None
    st = get_state(session_id)
    turns = st.get("turns") or []
    needle_l = needle.lower()
    for t in turns:
        content = t.get("content") or ""
        if needle_l in content.lower():
            s = content.strip().replace("\n", " ")
            return s if len(s) <= max_chars else s[: max_chars - 1] + "…"
    return None