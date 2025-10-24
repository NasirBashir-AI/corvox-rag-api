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

# ==== Field extractors ====
_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)
_PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{6,}\d")

_NAME_RE = re.compile(
    r"\b(?:my\s+name\s+is|i\s*'?m|i\s+am|this\s+is)\s+([A-Z][A-Za-z'.\- ]{1,40})\b",
    re.IGNORECASE,
)

_COMPANY_RE = re.compile(
    r"\b(?:company|business|organisation|organization|my\s+business\s+is)\s*(?:name|called|is)?\s*([A-Z][\w&'.\- ]{1,60})\b",
    re.IGNORECASE,
)

_TIME_RE = re.compile(
    r"\b(?:(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*|tomorrow|today)\b|\b\d{1,2}[:.]\d{2}\s*(?:am|pm)\b",
    re.IGNORECASE,
)

# ==== Ask gating ====
ASK_INTERVAL_TURNS = 2      # wait at least 2 turns between new asks
MATURITY_TURNS     = 3      # minimum turns before any unsolicited ask

ORDER = ["name", "company", "phone", "email", "time"]  # priority order


def extract_lead_fields(text: str) -> Dict[str, str]:
    """Extract name, email, phone, company, time from user text."""
    lead: Dict[str, str] = {}

    if m := _EMAIL_RE.search(text):
        lead["email"] = m.group(0).strip()

    if m := _PHONE_RE.search(text):
        lead["phone"] = m.group(0).strip()

    if m := _COMPANY_RE.search(text):
        lead["company"] = m.group(1).strip()

    if m := _NAME_RE.search(text):
        possible = m.group(1).strip()
        lead["name"] = " ".join(w.capitalize() for w in re.split(r"\s+", possible))

    if m := _TIME_RE.search(text):
        lead["time"] = m.group(0).strip()

    return lead


def update_lead_info(session_id: str, text: str) -> Dict[str, str]:
    """
    Merge any newly extracted fields from `text` into session lead slots.
    Never overwrite pre-existing values.
    """
    slots = get_lead_slots(session_id)
    found = extract_lead_fields(text)
    for k, v in found.items():
        if v and not slots.get(k):
            update_lead_slot(session_id, k, v)
            slots[k] = v
    return slots


def next_lead_question(
    session_id: str,
    turn_idx: int,
    user_intent: str = "",
    force: bool = False,
) -> Optional[str]:
    """
    Decide the next single lead-capture question, or None if we shouldn't ask now.
    Gating:
      - Wait at least ASK_INTERVAL_TURNS between asks unless force=True
      - Only ask for fields we don't have yet
      - Trigger asking only if user_intent is 'lead'/'pricing'/'contact' or after MATURITY_TURNS.
    """
    state = get_state(session_id) or {}
    last_ask_turn = int(state.get("last_ask_turn", -999))

    if not force and (turn_idx - last_ask_turn) < ASK_INTERVAL_TURNS:
        return None

    slots = get_lead_slots(session_id)
    missing = [f for f in ORDER if not slots.get(f)]

    if not missing:
        return None

    interest = user_intent in {"lead", "pricing", "contact"}
    matured = turn_idx >= MATURITY_TURNS
    if not (interest or matured or force):
        return None

    field = missing[0]
    set_state(session_id, last_asked_field=field, last_ask_turn=turn_idx)

    prompts = {
        "name": "What name should I use for you?",
        "company": "What’s your company name?",
        "phone": "Do you have a phone number I can note?",
        "email": "What’s the best email to reach you?",
        "time": "What day or time suits you for a quick discovery call?",
    }
    return prompts[field]


def is_ready_for_callback(slots: Dict[str, str]) -> bool:
    """Minimum details to pass to team for callback."""
    return bool(slots.get("name") and (slots.get("phone") or slots.get("email")) and slots.get("time"))


def format_user_details(slots: Dict[str, str]) -> str:
    parts = []
    for k in ORDER:
        v = slots.get(k)
        if v:
            parts.append(f"{k}: {v}")
    return "\n".join(parts) if parts else ""