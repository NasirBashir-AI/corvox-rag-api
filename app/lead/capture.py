from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from app.core.session_mem import get_state, set_state
from app.retrieval.leads import mark_stage, mark_done

# ---------- Lightweight extractors ----------
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
    if m:
        raw = m.group(2).strip()
    else:
        # Accept short single-line names like “Sam Patel”
        words = t.split()
        raw = t if 1 <= len(words) <= 5 else ""
    if not raw:
        return None
    parts = re.split(r"\s+", raw)
    return " ".join(p[:1].upper() + p[1:].lower() for p in parts)

# ---------- Helpers ----------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def in_progress(session_id: str) -> bool:
    st = get_state(session_id)
    stage = (st or {}).get("lead_stage")
    return bool(stage and stage != "done")

def start(session_id: str, kind: str = "callback") -> str:
    """
    Begin a lead capture flow. Stage -> 'name'
    Also persists an initial row with stage='name'.
    Micro-guard: if already done, don’t restart.
    """
    st = get_state(session_id) or {}
    if st.get("lead_stage") == "done":
        # Gentle reminder instead of restarting
        return "We’ve got your details on file already. If you’d like to update anything or need something else, just let me know."
    set_state(session_id, lead_stage="name", lead_kind=kind, lead_started_at=_now_iso())
    mark_stage(session_id, stage="name", source="chat")
    return "Great — I can arrange that. What’s your name?"

def take_turn(session_id: str, text: str) -> str:
    """
    Advance the state machine one step and persist at each transition.
    Returns the next question/confirmation to show the user.
    Stages: name -> contact -> time -> notes -> done
    """
    st = get_state(session_id) or {}
    stage = st.get("lead_stage") or "name"

    # Micro-guard: normalize any unexpected stage to "name"
    if stage not in {"name", "contact", "time", "notes", "done"}:
        stage = "name"
        set_state(session_id, lead_stage="name")
        mark_stage(session_id, stage="name")

    # --- NAME ---
    if stage == "name":
        n = harvest_name(text)
        if not n:
            return "Could you share your name (e.g., “I’m Sam Patel”)?"
        set_state(session_id, name=n, lead_stage="contact")
        mark_stage(session_id, stage="contact", name=n, source="chat")
        return f"Thanks, {n}. What’s the best phone number or email to reach you?"

    # --- CONTACT ---
    if stage == "contact":
        em = harvest_email(text)
        ph = harvest_phone(text)
        if em:
            set_state(session_id, email=em)
        if ph:
            set_state(session_id, phone=ph)

        if not (em or ph or st.get("email") or st.get("phone")):
            return "I didn’t catch a valid phone or email. Please share one of them."

        set_state(session_id, lead_stage="time")
        mark_stage(
            session_id,
            stage="time",
            email=em or st.get("email"),
            phone=ph or st.get("phone"),
        )
        return "When is a good time (and timezone) for us to contact you?"

    # --- TIME ---
    if stage == "time":
        pref = (text or "").strip()
        if not pref:
            return "What time works best (and your timezone)?"
        set_state(session_id, preferred_time=pref, lead_stage="notes")
        mark_stage(session_id, stage="notes", preferred_time=pref)
        return "Got it. Any extra context about your needs? (optional)"

    # --- NOTES ---
    if stage == "notes":
        notes = (text or "").strip()
        if notes:
            set_state(session_id, notes=notes)

        st = get_state(session_id) or {}
        # Persist final + mark done
        mark_done(
            session_id,
            name=st.get("name"),
            phone=st.get("phone"),
            email=st.get("email"),
            preferred_time=st.get("preferred_time"),
            notes=st.get("notes") or notes or None,
            done_at=_now_iso(),
        )
        set_state(session_id, lead_stage="done", lead_done_at=_now_iso())
        # Polite sign-off. The API can also send end_session=True (see main.py)
        return (
            "All set — I’ve logged your request and we’ll be in touch shortly. "
            "I’ll close this chat now. If you have more questions later, just start a new session. Take care!"
        )

    # --- DONE (or unknown) ---
    set_state(session_id, lead_stage="done")
    return "We’ve got your details noted. Anything else I can help with?"