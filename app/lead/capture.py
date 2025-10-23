# app/lead/capture.py
from __future__ import annotations
import re
from typing import Dict, Optional
from app.core.session_mem import get_state, set_state, get_lead_slots, update_lead_slot

# Regex extractors
_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.I)
_PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{6,}\d")
_NAME_RE = re.compile(r"\b(?:my\s+name\s+is|i\s*'?m|i\s+am|this\s+is)\s+([A-Z][A-Za-z'.\- ]{1,40})\b", re.I)
_COMPANY_RE = re.compile(r"\b(?:company|business|organisation|organization)\s*(?:name|called|is)?\s*([A-Z][\w&'.\- ]{1,60})\b", re.I)
_TIME_RE = re.compile(r"\b(?:\d{1,2}[:.]\d{2}\s*(?:am|pm)|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)

ASK_INTERVAL_TURNS = 2  # at least 2 turns apart

def extract_lead_fields(text: str) -> Dict[str, str]:
    lead: Dict[str, str] = {}
    if m := _EMAIL_RE.search(text): lead["email"] = m.group(0).strip()
    if m := _PHONE_RE.search(text): lead["phone"] = m.group(0).strip()
    if m := _COMPANY_RE.search(text): lead["company"] = m.group(1).strip()
    if m := _NAME_RE.search(text):
        possible = m.group(1).strip()
        lead["name"] = " ".join(w.capitalize() for w in re.split(r"\s+", possible))
    if m := _TIME_RE.search(text): lead["time"] = m.group(0).strip()
    return lead

def update_lead_info(session_id: str, text: str) -> Dict[str, str]:
    slots = get_lead_slots(session_id)
    found = extract_lead_fields(text)
    for k, v in found.items():
        if v and not slots.get(k):
            update_lead_slot(session_id, k, v)
            slots[k] = v
    return slots

def next_lead_question(session_id: str, turn_idx: int, user_intent: str = "") -> Optional[str]:
    """
    Decide next polite lead question.
    - Only ask if user shows buying intent or after 3+ turns.
    - Never repeat same question twice.
    """
    state = get_state(session_id) or {}
    last_ask_turn = int(state.get("last_ask_turn", -999))
    last_asked_field = state.get("last_asked_field", "")
    if (turn_idx - last_ask_turn) < ASK_INTERVAL_TURNS:
        return None

    slots = get_lead_slots(session_id)
    missing = [f for f in ("name", "company", "email", "phone", "time") if not slots.get(f)]
    if not missing:
        return None

    # Only ask if user is interested (lead/contact/pricing/info) or chat is matured
    salesy = user_intent in {"lead", "pricing", "contact", "info"}
    matured = turn_idx >= 4
    if not (salesy or matured):
        return None

    field = missing[0]
    if field == last_asked_field:
        return None  # avoid repeats

    set_state(session_id, last_asked_field=field, last_ask_turn=turn_idx)

    prompts = {
        "name": "May I have your name, please?",
        "company": "What’s your company name?",
        "email": "What’s the best email to reach you?",
        "phone": "Could you share a phone number I can note?",
        "time": "What day or time works best for a short discovery call?",
    }
    polite_prefix = ""
    if slots.get("name") and field == "company":
        polite_prefix = f"Perfect, {slots['name']} — got that. "
    elif slots.get("company") and field == "email":
        polite_prefix = f"Thanks, {slots.get('name','')}! "

    return polite_prefix + prompts[field]