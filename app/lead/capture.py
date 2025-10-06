# app/lead/capture.py
from __future__ import annotations
import re, uuid, json
from typing import Dict, Optional
from app.core.config import DB_URL
from app.core.utils import pg_cursor

EMAIL_RE  = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE  = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
NAME_RE   = re.compile(r"\b(i'?m|i am|this is)\s+([A-Za-z][A-Za-z\-\' ]{1,40})", re.I)

# in-memory flow per session
_SESS: Dict[str, Dict] = {}

def in_progress(session_id: str) -> bool:
    return session_id in _SESS

def start(session_id: str, kind: str = "callback") -> str:
    _SESS[session_id] = {
        "id": "LEAD-" + uuid.uuid4().hex[:8].upper(),
        "kind": kind,                      # 'callback' / 'demo' / 'consult'
        "stage": "name",                   # name -> contact -> time -> notes -> done
        "data": {}
    }
    return "Great — I can arrange that. What’s your name?"

def _extract_name(text: str) -> Optional[str]:
    m = NAME_RE.search(text)
    return (m.group(2).strip() if m else None) or (text.strip() if 1 <= len(text.split()) <= 5 else None)

def _extract_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None

def _extract_phone(text: str) -> Optional[str]:
    m = PHONE_RE.search(text)
    return m.group(0) if m else None

def take_turn(session_id: str, text: str) -> str:
    s = _SESS.get(session_id)
    if not s:
        return start(session_id)

    st = s["stage"]
    d  = s["data"]

    if st == "name":
        n = _extract_name(text)
        if not n:
            return "Could you share your name (e.g., “I’m Sam Patel”)?"
        d["name"] = n
        s["stage"] = "contact"
        return f"Thanks, {n}. What’s the best email or phone to reach you?"

    if st == "contact":
        em, ph = _extract_email(text), _extract_phone(text)
        if em: d["email"] = em
        if ph: d["phone"] = ph
        if not (em or ph):
            return "I didn’t catch a valid email or phone. Please share one of them."
        s["stage"] = "time"
        return "When is a good time (and timezone) for us to contact you?"

    if st == "time":
        d["time_pref"] = text.strip()
        s["stage"] = "notes"
        return "Got it. Any extra context about your needs? (optional)"

    if st == "notes":
        if text.strip():
            d["notes"] = text.strip()
        s["stage"] = "done"
        _save(session_id, s)
        lead_id = s["id"]
        _SESS.pop(session_id, None)
        return f"All set! I’ve logged your request ({lead_id}). We’ll contact you shortly. Anything else I can help with?"

    # fallback
    return "Let me just confirm—could you share your name?"

def _save(session_id: str, s: Dict) -> None:
    payload = {
        "lead_id": s["id"],
        "session_id": session_id,
        "kind": s["kind"],
        **s["data"]
    }
    sql = """
    CREATE TABLE IF NOT EXISTS corah_store.leads(
      id BIGSERIAL PRIMARY KEY,
      session_id TEXT,
      source TEXT DEFAULT 'chat',
      payload JSONB NOT NULL,
      created_at TIMESTAMPTZ DEFAULT now()
    );
    INSERT INTO corah_store.leads(session_id, payload)
    VALUES (%s, %s::jsonb);
    """
    with pg_cursor(DB_URL) as cur:
        cur.execute(sql, (session_id, json.dumps(payload)))