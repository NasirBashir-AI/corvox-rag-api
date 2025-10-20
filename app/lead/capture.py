# app/lead/capture.py
from __future__ import annotations
import re
from typing import Dict, Optional
from app.core.session_mem import (
    get_state as get_session,
    set_state as update_session,
)

# ---- Lead field extractors ----
_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)
_PHONE_RE = re.compile(r"\+?\d[\d\s\-]{7,}\d")
_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b")
_COMPANY_RE = re.compile(r"(?:company|business|organisation|organization)\s*(?:name|called|is)?\s*([A-Z][\w&\s]+)", re.IGNORECASE)
_TIME_RE = re.compile(r"(\b\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.)\b|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b)", re.IGNORECASE)

# ---- Ask gating ----
ASK_INTERVAL_TURNS = 2  # at least 2 turns between new asks

def extract_lead_fields(text: str) -> Dict[str, str]:
    """Extract name, email, phone, company, and time from user text."""
    lead: Dict[str, str] = {}
    if m := _EMAIL_RE.search(text):
        lead["email"] = m.group(0).strip()
    if m := _PHONE_RE.search(text):
        lead["phone"] = m.group(0).strip()
    if m := _COMPANY_RE.search(text):
        lead["company"] = m.group(1).strip()
    if m := _TIME_RE.search(text):
        lead["time"] = m.group(1).strip()
    # name as a fallback if context allows (avoid false positives)
    if not lead.get("name") and "my name" in text.lower():
        possible = re.sub(r"^.*my name is\s*", "", text, flags=re.I)
        lead["name"] = possible.split()[0].strip().title()
    return lead


def update_lead_info(session_id: str, text: str) -> Dict[str, str]:
    """Update session lead data from user text."""
    session = get_session(session_id)
    new_fields = extract_lead_fields(text)
    session["lead"].update(new_fields)
    update_session(session_id, session)
    return session["lead"]


def next_lead_question(session_id: str, turn_idx: int, user_intent: str = "") -> Optional[str]:
    """
    Return the next lead-capture question, or None if we shouldn’t ask yet.
    Gating logic ensures we don't interrogate too soon or repeat.
    """
    s = get_session(session_id)
    lead = s.get("lead", {})
    last_asked = s.get("last_asked_field")
    last_ask_turn = s.get("last_ask_turn", -ASK_INTERVAL_TURNS)

    # Too soon? Wait at least ASK_INTERVAL_TURNS since last ask.
    if turn_idx - last_ask_turn < ASK_INTERVAL_TURNS:
        return None

    # Determine missing fields in order of importance
    missing = [f for f in ["name", "company", "email", "phone", "time"] if not lead.get(f)]
    if not missing:
        return None

    # Trigger only if user shows intent (pricing/demo/contact etc.) or conversation matured
    if not user_intent and turn_idx < 3:
        return None

    next_field = missing[0]

    prompts = {
        "name": "May I have your name, please?",
        "company": "Could you tell me your company name?",
        "email": "What’s the best email address to reach you?",
        "phone": "Could you share your contact number?",
        "time": "When would be a convenient time for a quick call?",
    }

    s["last_asked_field"] = next_field
    s["last_ask_turn"] = turn_idx
    update_session(session_id, s)
    return prompts[next_field]


def summarize_lead(session_id: str) -> str:
    """Compact one-line summary for reporting or closure."""
    lead = get_session(session_id).get("lead", {})
    parts = []
    if n := lead.get("name"): parts.append(n)
    if c := lead.get("company"): parts.append(f"({c})")
    if e := lead.get("email"): parts.append(f"📧 {e}")
    if p := lead.get("phone"): parts.append(f"📞 {p}")
    if t := lead.get("time"): parts.append(f"🕒 {t}")
    return " ".join(parts) if parts else "—"