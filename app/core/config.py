# app/lead/capture.py
from __future__ import annotations
import re
from typing import Dict
from app.core.session_mem import get_lead_slots, update_lead_slot

_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d[\s\-()]*){7,}\d\b")
_NAME_RE  = re.compile(r"\b(?:my\s+name\s+is|i\s*'?m|i\s+am|this\s+is)\s+([A-Z][A-Za-z'.\- ]{1,60})\b", re.IGNORECASE)
_COMPANY_RE = re.compile(r"\b(?:company|business|organisation|organization)\s*(?:name|called|is)?\s*([A-Z][\w&'.\- ]{1,80})\b", re.IGNORECASE)
_TIME_RE = re.compile(r"\b(?:tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}[:.]\d{2}\s*(am|pm))\b", re.IGNORECASE)

def extract_lead_fields(text: str) -> Dict[str, str]:
    lead: Dict[str, str] = {}
    if m := _EMAIL_RE.search(text):   lead["email"] = m.group(0).strip()
    if m := _PHONE_RE.search(text):   lead["phone"] = m.group(0).strip()
    if m := _COMPANY_RE.search(text): lead["company"] = m.group(1).strip()
    if m := _NAME_RE.search(text):
        possible = m.group(1).strip()
        lead["name"] = " ".join(w.capitalize() for w in re.split(r"\s+", possible))
    if m := _TIME_RE.search(text):    lead["preferred_time"] = m.group(0).strip()
    return lead

def update_lead_info(session_id: str, text: str) -> Dict[str, str]:
    slots = get_lead_slots(session_id)
    found = extract_lead_fields(text or "")
    for k, v in found.items():
        if v and not slots.get(k):
            update_lead_slot(session_id, k, v)
            slots[k] = v
    return slots