# app/lead/capture.py
from __future__ import annotations

import re
import uuid
from typing import Dict, Optional

from app.core.config import DB_URL
from app.core.utils import pg_cursor
from app.core.session_mem import get_state, set_state


# -----------------------------
# Simple extractors (harvesters)
# -----------------------------

EMAIL_RE  = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE  = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
NAME_RE   = re.compile(r"\b(i'?m|i am|this is)\s+([A-Za-z][A-Za-z\-\' ]{1,40})", re.I)

def harvest_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text or "")
    return m.group(0) if m else None

def harvest_phone(text: str) -> Optional[str]:
    m = PHONE_RE.search(text or "")
    return m.group(0).strip() if m else None

def harvest_name(text: str) -> Optional[str]:
    t = (text or "").strip()
    m = NAME_RE.search(t)
    if not m:
        return None
    # group(2) is the actual name (group 1 is "i'm / i am / this is")
    raw = m.group(2).strip()
    parts = re.split(r"\s+", raw)
    return " ".join(p[:1].upper() + p[1:].lower() for p in parts)


# -----------------------------
# Lead flow (short-term memory)
# -----------------------------
# We keep the flow state per session inside session_mem, not a module global,
# so multiple workers/processes see the same in-memory state.

_LEAD_KEY = "lead_flow"   # entire dict stored under this key in session state


def _get_lead(session_id: str) -> Optional[Dict]:
    st = get_state(session_id) or {}
    return st.get(_LEAD_KEY)


def _set_lead(session_id: str, lead: Dict) -> None:
    st = get_state(session_id) or {}
    st[_LEAD_KEY] = lead
    set_state(session_id, **st)


def _clear_lead(session_id: str) -> None:
    st = get_state(session_id) or {}
    if _LEAD_KEY in st:
        st.pop(_LEAD_KEY, None)
        set_state(session_id, **st)


def in_progress(session_id: str) -> bool:
    """Is a lead-capture conversation currently in progress for this session?"""
    return _get_lead(session_id) is not None


def start(session_id: str, kind: str = "callback") -> str:
    """Start a new lead flow: name -> contact -> time -> notes -> done."""
    lead = {
        "id": "LEAD-" + uuid.uuid4().hex[:8].upper(),
        "kind": kind,                         # 'callback' / 'demo' / 'consult'
        "stage": "name",                      # name -> contact -> time -> notes
        "data": {},                           # name / phone / email / preferred_time / notes
    }
    _set_lead(session_id, lead)
    return "Great — I can arrange that. What’s your name?"


def _extract_name(text: str) -> Optional[str]:
    m = NAME_RE.search(text or "")
    if m:
        return m.group(2).strip()
    # fallback: a short single-line name without the “I'm …” pattern
    t = (text or "").strip()
    if 1 <= len(t.split()) <= 5 and all(ch.isalpha() or ch in " -'" for ch in t):
        return t
    return None


def _extract_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text or "")
    return m.group(0) if m else None


def _extract_phone(text: str) -> Optional[str]:
    m = PHONE_RE.search(text or "")
    return m.group(0).strip() if m else None


def take_turn(session_id: str, text: str) -> str:
    """
    Advance the multi-turn capture flow using the current message text.
    Stages: name -> contact -> time -> notes -> done
    """
    lead = _get_lead(session_id)
    if not lead:
        # If the flow vanished, restart gently.
        return start(session_id)

    stage = lead["stage"]
    data  = lead["data"]

    # 1) Name
    if stage == "name":
        n = _extract_name(text)
        if not n:
            return "Could you share your name (e.g., “I’m Sam Patel”)?"
        data["name"] = n
        lead["stage"] = "contact"
        _set_lead(session_id, lead)
        # Ask for phone first for callbacks; email is accepted too.
        return f"Thanks, {n}. What’s the best **phone number** for you? (You can share an email instead if you prefer.)"

    # 2) Contact
    if stage == "contact":
        em, ph = _extract_email(text), _extract_phone(text)
        if em:
            data["email"] = em
        if ph:
            data["phone"] = ph
        if not (em or ph):
            return "I didn’t catch a valid phone or email. Please share one of them."
        lead["stage"] = "time"
        _set_lead(session_id, lead)
        return "When is a good time (and timezone) for us to contact you?"

    # 3) Time preference
    if stage == "time":
        pref = (text or "").strip()
        if pref:
            data["preferred_time"] = pref
        lead["stage"] = "notes"
        _set_lead(session_id, lead)
        return "Got it. Any extra context about your needs? (optional)"

    # 4) Notes (final)
    if stage == "notes":
        note = (text or "").strip()
        if note:
            data["notes"] = note

        # Persist a structured lead row
        try:
            save_lead(
                session_id=session_id,
                name=data.get("name"),
                phone=data.get("phone"),
                email=data.get("email"),
                preferred_time=data.get("preferred_time"),
                notes=data.get("notes"),
                source="chat",
            )
        finally:
            lead_id = lead["id"]
            _clear_lead(session_id)

        return f"All set! I’ve logged your request ({lead_id}). We’ll contact you shortly. Anything else I can help with?"

    # Fallback
    return "Let me just confirm—could you share your name?"


# -----------------------------
# DB persistence (short-term path)
# -----------------------------

def save_lead(
    session_id: str,
    name: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    preferred_time: Optional[str] = None,
    notes: Optional[str] = None,
    source: str = "chat",
) -> None:
    """
    Insert a structured lead row. This uses a lightweight DDL guard so it works
    even if the table hasn't been created yet.
    """
    sql = """
    CREATE TABLE IF NOT EXISTS corah_store.leads(
        id BIGSERIAL PRIMARY KEY,
        session_id     TEXT,
        name           TEXT,
        phone          TEXT,
        email          TEXT,
        preferred_time TEXT,
        notes          TEXT,
        source         TEXT DEFAULT 'chat',
        created_at     TIMESTAMPTZ DEFAULT now(),
        updated_at     TIMESTAMPTZ DEFAULT now()
    );

    INSERT INTO corah_store.leads
        (session_id, name, phone, email, preferred_time, notes, source)
    VALUES
        (%s, %s, %s, %s, %s, %s, %s);
    """
    with pg_cursor(DB_URL) as cur:
        cur.execute(
            sql,
            (
                session_id,
                name,
                phone,
                email,
                preferred_time,
                notes,
                source,
            ),
        )