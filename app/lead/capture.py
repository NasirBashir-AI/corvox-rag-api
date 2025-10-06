# app/lead/capture.py

from typing import Optional, Dict, Any
from app.core.session_mem import get_state, set_state

# You already have these in this file:
# EMAIL_RE, PHONE_RE, NAME_RE
# def harvest_email(text: str) -> Optional[str]: ...
# def harvest_phone(text: str) -> Optional[str]: ...
# def harvest_name(text: str) -> Optional[str]: ...

FLOW_KEY = "lead_flow"      # holds "idle"|"active"
STAGE_KEY = "lead_stage"    # "lead_name"|"lead_contact"|"lead_time"|"lead_notes"

def _state(sid: str) -> Dict[str, Any]:
    return get_state(sid) or {}

def _set(sid: str, **kv) -> None:
    cur = _state(sid)
    cur.update(kv)
    set_state(sid, **cur)

def in_progress(session_id: str) -> bool:
    st = _state(session_id)
    return st.get(FLOW_KEY) == "active" and st.get(STAGE_KEY) in {
        "lead_name", "lead_contact", "lead_time", "lead_notes"
    }

def start(session_id: str, kind: str = "callback") -> str:
    """
    Start or resume the lead flow from the *first missing piece*.
    """
    st = _state(session_id)
    name  = st.get("name")
    phone = st.get("phone")
    email = st.get("email")

    _set(session_id, **{FLOW_KEY: "active"})  # mark active

    if not name:
        _set(session_id, **{STAGE_KEY: "lead_name"})
        return "Got it. May I take your name?"
    if not (phone or email):
        _set(session_id, **{STAGE_KEY: "lead_contact"})
        pre = f"Thanks {name}. " if name else ""
        return pre + "What’s the best phone number for a callback? You can also share an email."
    # (Optional) preferred time
    if not st.get("preferred_time"):
        _set(session_id, **{STAGE_KEY: "lead_time"})
        return "When is a good time for us to contact you?"
    # (Optional) notes/context
    if not st.get("notes"):
        _set(session_id, **{STAGE_KEY: "lead_notes"})
        return "Any extra context about your needs?"

    # If everything is already present, we’re done.
    _set(session_id, **{FLOW_KEY: "idle", STAGE_KEY: None})
    return "Thanks, I’ve got the details. We’ll be in touch shortly."

def take_turn(session_id: str, user_text: str) -> str:
    """
    Handle the current lead stage; harvest any details we can from the message,
    update memory, then decide the next prompt.
    """
    st = _state(session_id)

    # Always try to harvest on every turn (so we don’t re-ask)
    name  = harvest_name(user_text)   or st.get("name")
    phone = harvest_phone(user_text)  or st.get("phone")
    email = harvest_email(user_text)  or st.get("email")

    changed = {}
    if name and name != st.get("name"):
        changed["name"] = name
    if phone and phone != st.get("phone"):
        changed["phone"] = phone
    if email and email != st.get("email"):
        changed["email"] = email
    if changed:
        _set(session_id, **changed)

    stage = _state(session_id).get(STAGE_KEY)

    # Progress the flow based on what we have now
    if stage == "lead_name":
        if not _state(session_id).get("name"):
            return "Got it. May I take your name?"
        # advance
        _set(session_id, **{STAGE_KEY: "lead_contact"})
        return f"Thanks {name}. What’s the best phone number for a callback? You can also share an email."

    if stage == "lead_contact":
        have_contact = _state(session_id).get("phone") or _state(session_id).get("email")
        if not have_contact:
            return "What’s the best phone number for a callback? You can also share an email."
        _set(session_id, **{STAGE_KEY: "lead_time"})
        return "When is a good time for us to contact you?"

    if stage == "lead_time":
        # You can add a real time parser later; for now just store the raw text.
        if user_text.strip():
            _set(session_id, preferred_time=user_text.strip())
        if not _state(session_id).get("preferred_time"):
            return "When is a good time for us to contact you?"
        _set(session_id, **{STAGE_KEY: "lead_notes"})
        return "Any extra context about your needs?"

    if stage == "lead_notes":
        if user_text.strip():
            _set(session_id, notes=user_text.strip())
        _set(session_id, **{FLOW_KEY: "idle", STAGE_KEY: None})
        person = _state(session_id).get("name")
        if _state(session_id).get("phone") and _state(session_id).get("email"):
            return f"Thanks {person}. I’ve noted your phone and email. We’ll be in touch shortly."
        if _state(session_id).get("phone"):
            return f"Thanks {person}. I’ve got your phone number. We’ll call you shortly."
        if _state(session_id).get("email"):
            return f"Thanks {person}. I’ve noted your email."
        return "Thanks, I’ve got the details. We’ll be in touch shortly."

    # If we ever land here, re-run start() to pick the right step.
    return start(session_id, kind="callback")