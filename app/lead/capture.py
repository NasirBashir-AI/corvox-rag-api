from __future__ import annotations
import re
from typing import Dict, Optional
from app.core.session_mem import get_lead_slots, update_lead_slot

_EMAIL_RE  = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)
_PHONE_RE  = re.compile(r"\b(?:\+?\d[\s\-()]*){7,}\d\b")
_NAME_RE   = re.compile(r"\b(?:my\s+name\s+is|i\s*'?m|i\s+am|this\s+is)\s+([A-Z][A-Za-z'.\- ]{1,40})\b", re.IGNORECASE)
_COMPANY_RE= re.compile(r"\b(?:company|business|organisation|organization)\s*(?:name|called|is)?\s*([A-Z][\w&'.\- ]{1,60})\b",
                        re.IGNORECASE)
_TIME_RE   = re.compile(r"\b(?:(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*|tomorrow|today|\d{1,2}[:.]\d{2}\s*(?:am|pm))\b",
                        re.IGNORECASE)

def update_lead_info(session_id: str, text: str) -> Dict[str, str]:
    slots = get_lead_slots(session_id)
    if m := _EMAIL_RE.search(text):   update_lead_slot(session_id, "email", m.group(0))
    if m := _PHONE_RE.search(text):   update_lead_slot(session_id, "phone", m.group(0))
    if m := _NAME_RE.search(text):    update_lead_slot(session_id, "name",  " ".join(w.capitalize() for w in m.group(1).split()))
    if m := _COMPANY_RE.search(text): update_lead_slot(session_id, "company", m.group(1))
    if m := _TIME_RE.search(text):    update_lead_slot(session_id, "time", m.group(0))
    return get_lead_slots(session_id)

def next_lead_question(slots: Dict[str,str], allow: bool) -> Optional[str]:
    if not allow: return None
    order = ["name","company","email","phone","time"]
    prompts = {
        "name":"What name should I use for you?",
        "company":"What’s your company name?",
        "email":"What’s the best email to reach you?",
        "phone":"Do you have a phone number I can note?",
        "time":"What day or time suits you for a quick discovery call?"
    }
    for f in order:
        if not slots.get(f):
            return prompts[f]
    return None