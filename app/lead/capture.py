# app/lead/capture.py
from __future__ import annotations

import re
from typing import Dict, Optional

from app.core.session_mem import (
    get_state,
    set_state,
    get_lead_slots,
    update_lead_slot,
)

# ---------- Lead field extractors ----------
_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)
_PHONE_RE = re.compile(r"\+?\d[\d\s\-]{6,}\d")
_NAME_RE  = re.compile(r"\b(?:I'?m|I am|name (?:is|=)|it'?s|call(?:ed)?)[:\s]*([A-Z][\w\s']+)\b", re.IGNORECASE)
_COMPANY_RE = re.compile(r"\b(?:company|business|organisation|organization)\s*(?:name|called|is)?\s*([A-Z][\w\s]+)", re.IGNORECASE)
_TIME_RE = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)\b|\b(mon|tue|wed|thu|fri|sat|sun)\b", re.IGNORECASE)

# ---------- Ask gating ----------
ASK_INTERVAL_TURNS = 2  # minimum turns between new asks


def _ensure_session(session_id: str) -> Dict:
    """Create default session scaffolding if missing."""
    st = get_state(session_id) or {}
    st.setdefault("lead", {})
    st.setdefault("turn_idx", 0)
    st.setdefault("last_ask_turn", -999)
    st.setdefault("last_ask_field", "")
    set_state(session_id, **st)
    return st


def extract_lead_fields(text: str) -> Dict[str, str]:
    """Extract name, email, phone, company, time from free text."""
    lead: Dict[str, str] = {}

    if m := _EMAIL_RE.search(text):
        lead["email"] = m.group(0).strip()

    if m := _PHONE_RE.search(text):
        lead["phone"] = m.group(0).strip()

    if m := _COMPANY_RE.search(text):
        lead["company"] = m.group(1).strip()

    if m := _NAME_RE.search(text):
        possible = (m.group(1) or "").strip()
        # Keep first token as conservative "name"
        if possible:
            lead["name"] = possible.split()[0].title()

    if m := _TIME_RE.search(text):
        lead["time"] = m.group(0).strip()

    return lead


def update_lead_info(session_id: str, text: str) -> Dict[str, str]:
    """Update session with any fields found in user text."""
    _ensure_session(session_id)
    found = extract_lead_fields(text)
    for k, v in found.items():
        update_lead_slot(session_id, k, v)
    # return the latest lead snapshot
    return get_lead_slots(session_id)


def next_lead_question(session_id: str, turn_idx: int, user_intent: str = "") -> Optional[str]:
    """
    Decide the next field to ask for (or None). Very conservative:
    - Only one ask every ASK_INTERVAL_TURNS.
    - Ask in priority: name -> company -> email -> phone -> time.
    - If user intent indicates pricing/demo/contact, we allow an ask.
    """
    st = _ensure_session(session_id)
    last_ask_turn = st.get("last_ask_turn", -999)
    last_ask_field = st.get("last_ask_field", "")

    if turn_idx - last_ask_turn < ASK_INTERVAL_TURNS:
        return None

    lead = get_lead_slots(session_id) or {}
    missing_order = [f for f in ("name", "company", "email", "phone", "time") if not lead.get(f)]
    if not missing_order:
        return None

    # Gate by intent (loosely)
    lowered = (user_intent or "").lower()
    allow = any(x in lowered for x in ("price", "pricing", "demo", "quote", "trial", "contact"))
    # Also allow once the conversation is slightly mature
    allow = allow or (turn_idx >= 2)
    if not allow:
        return None

    ask_field = missing_order[0]
    st["last_ask_turn"] = turn_idx
    st["last_ask_field"] = ask_field
    set_state(session_id, **st)

    prompts = {
        "name":    "What name should we use for you?",
        "company": "What’s your company name?",
        "email":   "What’s the best email to reach you?",
        "phone":   "What’s the best phone number?",
        "time":    "Is there a good time we should plan for a quick chat?",
    }
    return prompts.get(ask_field, None)


# ===========================
# Legacy API expected by main
# ===========================

def in_progress(session_id: str) -> bool:
    """Return True if we have a session record."""
    return get_state(session_id) is not None


def start(session_id: str) -> None:
    """Ensure session structure exists."""
    _ensure_session(session_id)


def take_turn(session_id: str, turn_idx: int, user_text: str, user_intent: str = "") -> Dict[str, Optional[str]]:
    """
    Legacy orchestration shim:
      - Update lead info from the user's free text
      - Decide next question (or None)
      - Return a small dict the old main.py understands
    """
    _ensure_session(session_id)
    update_lead_info(session_id, user_text)
    ask = next_lead_question(session_id, turn_idx, user_intent=user_intent)
    return {
        "ask": ask,
        "lead": get_lead_slots(session_id),
    }


def harvest_email(session_id: str, value: str) -> None:
    _ensure_session(session_id)
    if value:
        update_lead_slot(session_id, "email", value.strip())


def harvest_phone(session_id: str, value: str) -> None:
    _ensure_session(session_id)
    if value:
        update_lead_slot(session_id, "phone", value.strip())


def harvest_name(session_id: str, value: str) -> None:
    _ensure_session(session_id)
    if value:
        update_lead_slot(session_id, "name", value.strip().title())


def harvest_company(session_id: str, value: str) -> None:
    _ensure_session(session_id)
    if value:
        update_lead_slot(session_id, "company", value.strip())


def harvest_time(session_id: str, value: str) -> None:
    _ensure_session(session_id)
    if value:
        update_lead_slot(session_id, "time", value.strip())