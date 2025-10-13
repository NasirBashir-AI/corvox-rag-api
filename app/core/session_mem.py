# app/core/session_mem.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from app.core.config import (
    DB_URL,
    CTA_COOLDOWN_TURNS,
)
from app.core.utils import pg_cursor

# -------------------------------------------------------------------
# State shape (single JSONB per session)
# -------------------------------------------------------------------
# {
#   "name": null, "phone": null, "email": null, "preferred_time": null, "timezone": null,
#   "lead_stage": null,              # name|contact|time|notes|done
#   "lead_started_at": null,
#   "lead_done_at": null,
#   "lead_just_done": false,
#   "asked_for_name_at": null,
#   "asked_for_contact_at": null,
#   "asked_for_time_at": null,
#   "asked_for_notes_at": null,
#   "last_asked": null,              # one of: name|contact|time|notes
#   "current_topic": null,           # short phrase: "WhatsApp chatbot for jewellery"
#   "session_summary": "",           # rolling text summary
#   "cta_attempts": 0,               # how many CTAs asked in session
#   "cta_last_turn_idx": null,       # assistant turn index of last CTA asked
#   "turn_index": 0,                 # monotonically increasing turn counter
#   "turns": [ {role:"user|assistant", content:"...", ts:"ISO"} ]  # rolling (last 20–30)
# }
# -------------------------------------------------------------------

_DEF_STATE: Dict[str, Any] = {
    "name": None,
    "phone": None,
    "email": None,
    "preferred_time": None,
    "timezone": None,

    "lead_stage": None,
    "lead_started_at": None,
    "lead_done_at": None,
    "lead_just_done": False,

    "asked_for_name_at": None,
    "asked_for_contact_at": None,
    "asked_for_time_at": None,
    "asked_for_notes_at": None,
    "last_asked": None,

    "current_topic": None,
    "session_summary": "",

    "cta_attempts": 0,
    "cta_last_turn_idx": None,

    "turn_index": 0,
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

# -------------------------------------------------------------------
# Low-level store helpers
# -------------------------------------------------------------------

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

# -------------------------------------------------------------------
# Turns & windowed memory
# -------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def append_turn(session_id: str, role: str, content: str, keep_last: int = 24) -> None:
    st = get_state(session_id)
    turns: List[Dict[str, Any]] = st.get("turns") or []
    turns.append({"role": role, "content": content, "ts": _now_iso()})
    if len(turns) > keep_last:
        turns = turns[-keep_last:]
    st["turns"] = turns
    # bump the turn index on each append
    st["turn_index"] = int(st.get("turn_index") or 0) + 1
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

# -------------------------------------------------------------------
# Asked markers & cooldown
# -------------------------------------------------------------------

def mark_asked(session_id: str, field: str) -> None:
    """
    Stamp asked_for_<field>_at and last_asked.
    Accepts field in {"name","contact","time","notes"}.
    """
    field = (field or "").strip().lower()
    key = None
    if field in ("name", "contact", "time", "notes"):
        key = f"asked_for_{field}_at"
    if not key:
        return
    now_iso = _now_iso()
    st = get_state(session_id)
    st[key] = now_iso
    st["last_asked"] = field
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_UPSERT, {"sid": session_id, "state": json.dumps(st)})

def recently_asked(session_id: str, field: str, cooldown_seconds: int) -> bool:
    field = (field or "").strip().lower()
    key = None
    if field in ("name", "contact", "time", "notes"):
        key = f"asked_for_{field}_at"
    if not key:
        return False
    st = get_state(session_id)
    iso = st.get(key)
    if not iso:
        return False
    try:
        last = datetime.fromisoformat(iso)
        now = datetime.now(timezone.utc)
        return (now - last).total_seconds() < max(1, int(cooldown_seconds))
    except Exception:
        return False

# -------------------------------------------------------------------
# CTA cool-down helpers
# -------------------------------------------------------------------

def can_ask_cta(session_id: str) -> bool:
    """
    Respect CTA_COOLDOWN_TURNS between assistant CTAs.
    """
    st = get_state(session_id)
    last_idx = st.get("cta_last_turn_idx")
    if last_idx is None:
        return True
    current_idx = int(st.get("turn_index") or 0)
    return (current_idx - int(last_idx)) >= int(CTA_COOLDOWN_TURNS)

def stamp_cta(session_id: str) -> None:
    st = get_state(session_id)
    st["cta_attempts"] = int(st.get("cta_attempts") or 0) + 1
    st["cta_last_turn_idx"] = int(st.get("turn_index") or 0)
    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_UPSERT, {"sid": session_id, "state": json.dumps(st)})

# -------------------------------------------------------------------
# Summary & topic updater (lightweight, rule-based)
# -------------------------------------------------------------------

# Tiny heuristics to pull signals from the latest user text
_INDICATORS = {
    "industries": {
        "fashion": ["fashion", "apparel", "clothing"],
        "jewellery": ["jewelry", "jewellery", "jewel"],
        "accountancy": ["accountancy", "accountant", "bookkeeping"],
        "education": ["student", "school", "university", "college"],
        "real_estate": ["estate", "property", "letting", "realtor"],
        "ecommerce": ["ecommerce", "online store", "shopify", "woocommerce"],
    },
    "goals": {
        "lead_capture": ["lead", "callback", "call back", "book a call", "contact me"],
        "customer_service": ["customer service", "support", "helpdesk", "chatbot", "whatsapp"],
        "automation": ["automate", "automation", "agent", "ai team", "ai department"],
        "pricing": ["pricing", "cost", "how much", "price"],
    }
}

def _first_hit(text: str, buckets: Dict[str, List[str]]) -> Optional[str]:
    tl = text.lower()
    for label, kws in buckets.items():
        for kw in kws:
            if kw in tl:
                return label
    return None

def _maybe_topic_from(text: str) -> Optional[str]:
    tl = (text or "").lower().strip()
    if not tl:
        return None
    # simple topic candidates
    if "whatsapp" in tl and "chatbot" in tl:
        return "WhatsApp chatbot"
    if "chatbot" in tl:
        return "Chatbot"
    if "reconciliation" in tl or "reconcile" in tl:
        return "Payments reconciliation"
    if "inventory" in tl or "stock" in tl:
        return "Inventory & stock answers"
    if "pricing" in tl or "cost" in tl:
        return "Pricing"
    return None

def update_summary(session_id: str, last_user_text: str) -> Dict[str, Any]:
    """
    Update session_summary and current_topic opportunistically from the last user message.
    - Keeps current_topic stable on backchannels ("yes", "sure", "tell me more").
    - Writes simple labeled summary lines.
    """
    st = get_state(session_id)
    txt = (last_user_text or "").strip()
    if not txt:
        return st

    # Infer industry & goal (sticky once found)
    industry = st.get("industry")
    goal = st.get("goal")

    ihit = _first_hit(txt, _INDICATORS["industries"])
    ghit = _first_hit(txt, _INDICATORS["goals"])

    if not industry and ihit:
        st["industry"] = ihit
    if ghit:
        st["goal"] = ghit if not goal else goal

    # Topic handling: only overwrite on NEW signal, not backchannel
    backchannels = set(["yes","sure","ok","okay","yep","yup","sounds good","go ahead","tell me more","please continue","continue"])
    if txt.lower() not in backchannels:
        topic = _maybe_topic_from(txt)
        if topic:
            st["current_topic"] = topic
        elif not st.get("current_topic") and industry:
            # seed with a generic but relevant topic if empty
            st["current_topic"] = "AI for " + industry.replace("_"," ")

    # Update summary (short, single paragraph)
    lined = []
    if st.get("industry"): lined.append(f"industry={st['industry']}")
    if st.get("goal"): lined.append(f"goal={st['goal']}")
    if st.get("current_topic"): lined.append(f"topic={st['current_topic']}")
    # user details (only if present)
    if st.get("name"): lined.append(f"name={st['name']}")
    if st.get("phone"): lined.append("phone=on-file")
    if st.get("email"): lined.append("email=on-file")
    if st.get("preferred_time"): lined.append(f"time={st['preferred_time']}")
    st["session_summary"] = " | ".join(lined)

    with pg_cursor(DB_URL) as cur:
        cur.execute(SQL_UPSERT, {"sid": session_id, "state": json.dumps(st)})
    return st

# -------------------------------------------------------------------
# Convenience for orchestrator / generator
# -------------------------------------------------------------------

def recent_turn_text(session_id: str, n: int = 6) -> str:
    """
    Return a compact recent-turns text block (user/assistant pairs), used in context.
    """
    turns = recent_turns(session_id, n=n)
    lines: List[str] = []
    for t in turns:
        role = t.get("role")
        content = (t.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)

def set_current_topic(session_id: str, topic: Optional[str], overwrite: bool = True) -> None:
    if not topic:
        return
    st = get_state(session_id)
    if overwrite or not st.get("current_topic"):
        set_state(session_id, current_topic=topic)