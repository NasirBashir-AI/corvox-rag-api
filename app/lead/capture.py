# app/lead/capture.py
from __future__ import annotations
import re
from typing import Dict, Optional
from app.core.session_mem import get_state, set_state, get_lead_slots, update_lead_slot

_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d[\s\-()]*){7,}\d\b")
_NAME_RE  = re.compile(r"\b(?:my\s+name\s+is|i\s*'?m|i\s+am|this\s+is)\s+([A-Z][A-Za-z'.\- ]{1,40})\b", re.IGNORECASE)
_COMPANY_RE = re.compile(r"\b(?:company|business|organisation|organization)\s*(?:name|called|is)?\s*([A-Z][\w&'.\- ]{1,60})\b", re.IGNORECASE)
_TIME_RE = re.compile(r"\b(?:\d{1,2}[:.]\d{2}\s*(?:am|pm)|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE)

ASK_INTERVAL_TURNS = 2  # min turns between asks

def extract_lead_fields(text: str) -> Dict[str, str]:
    lead: Dict[str, str] = {}
    if m := _EMAIL_RE.search(text):   lead["email"] = m.group(0).strip()
    if m := _PHONE_RE.search(text):   lead["phone"] = m.group(0).strip()
    if m := _COMPANY_RE.search(text): lead["company"] = m.group(1).strip()
    if m := _NAME_RE.search(text):
        possible = m.group(1).strip()
        lead["name"] = " ".join(w.capitalize() for w in re.split(r"\s+", possible))
    if m := _TIME_RE.search(text):    lead["time"] = m.group(0).strip()
    return lead

def update_lead_info(session_id: str, text: str) -> Dict[str, str]:
    slots = get_lead_slots(session_id)
    found = extract_lead_fields(text)
    for k, v in found.items():
        if v and not slots.get(k):
            update_lead_slot(session_id, k, v)
            slots[k] = v
    return slots

def next_lead_question(session_id: str, turn_idx: int, user_intent: str = "", ask_early: bool = False) -> Optional[str]:
    st = get_state(session_id) or {}
    last_ask_turn = int(st.get("last_ask_turn", -999))
    last_asked = st.get("last_asked_field", "")

    if (turn_idx - last_ask_turn) < ASK_INTERVAL_TURNS:
        return None

    slots = get_lead_slots(session_id)
    missing = [f for f in ("name","company","email","phone","time") if not slots.get(f)]
    if not missing:
        return None

    # Ask only when it makes sense: explicit salesy intent or later in the flow
    salesy = user_intent in {"pricing","lead","contact","services"}
    matured = turn_idx >= (1 if ask_early else 3)
    if not (salesy or matured):
        return None

    field = missing[0]
    if field == last_asked:
        return None

    set_state(session_id, last_asked_field=field, last_ask_turn=turn_idx)
    prompts = {
        "name": "What name should I use for you?",
        "company": "What’s your company name?",
        "email": "What’s the best email to reach you?",
        "phone": "Do you have a phone number I can note?",
        "time": "What day or time suits you for a quick discovery call?",
    }
    return prompts[field]