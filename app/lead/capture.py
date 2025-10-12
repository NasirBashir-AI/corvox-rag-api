# app/lead/capture.py
from __future__ import annotations

import os
import re
import json
from typing import Optional, Dict

from app.core.session_mem import get_state, set_state, mark_asked, recently_asked
from app.retrieval.leads import mark_stage, mark_done

# Defensive import of cooldown knobs
try:
    from app.core.config import (
        ASK_COOLDOWN_NAME_SECS,
        ASK_COOLDOWN_PHONE_SECS,
        ASK_COOLDOWN_EMAIL_SECS,
        ASK_COOLDOWN_TIME_SECS,
        ASK_COOLDOWN_NOTES_SECS,
    )
except Exception:
    ASK_COOLDOWN_NAME_SECS = 45
    ASK_COOLDOWN_PHONE_SECS = 45
    ASK_COOLDOWN_EMAIL_SECS = 45
    ASK_COOLDOWN_TIME_SECS = 45
    ASK_COOLDOWN_NOTES_SECS = 45

CONTACT_COOLDOWN_SECS = max(ASK_COOLDOWN_PHONE_SECS, ASK_COOLDOWN_EMAIL_SECS)

# ===== Extraction (fast) =====
EMAIL_RE  = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE  = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")

NAME_CUE_RE = re.compile(
    r"\b(i'?m|i am|this is|my name is|name is|name\s*:|it'?s)\s+([A-Za-z][A-Za-z\-\' ]{1,40})",
    re.I,
)

STOP_WORDS = {
    "hi","hello","hey","ok","okay","thanks","thank","please",
    "email","phone","number","call","whatsapp","website",
    "price","pricing","cost","service","help",
    "what","why","how","when","where","who","can","could","would","does","do",
}

def _normalize_person_name(raw: str) -> str:
    parts = re.split(r"\s+", (raw or "").strip())
    fixed = []
    for p in parts:
        if not p: continue
        fixed.append(p[:1].upper() + p[1:].lower())
    return " ".join(fixed)


def harvest_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text or "")
    return m.group(0) if m else None

def harvest_name(text: str) -> Optional[str]:
    """
    Best-effort person name from free text.
    Accepts either a cue (“my name is…”) or a clean short name by itself.
    Returns normalized name (e.g., 'nasir' -> 'Nasir') or None.
    """
    if not text:
        return None
    t = text.strip()

    # Try explicit cue first
    m = NAME_CUE_RE.search(t)
    candidate = (m.group(2).strip() if m else None)

    # Otherwise accept a plain short name reply (e.g., "nasir")
    if not candidate and _looks_like_person_name(t):
        candidate = t

    if not candidate:
        return None

    nm = _normalize_person_name(candidate)
    return nm if _looks_like_person_name(nm) else None

def harvest_phone(text: str) -> Optional[str]:
    m = PHONE_RE.search(text or "")
    return m.group(0).strip() if m else None


def _looks_like_person_name(s: str) -> bool:
    if not s: return False
    s = s.strip()
    if len(s) < 2 or len(s) > 40: return False
    if any(ch.isdigit() for ch in s): return False
    if any(x in s.lower() for x in ("@", "http", "www")): return False
    toks = [t for t in s.split() if t]
    if not (1 <= len(toks) <= 4): return False
    if sum(1 for t in toks if t.lower() in STOP_WORDS) >= max(1, len(toks) - 1):
        return False
    return all(re.fullmatch(r"[A-Za-z][A-Za-z\-']*", t) for t in toks)


def _maybe_update_name_from(text: str, session_id: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    # allow only if there's a cue OR it's a pure short name
    has_cue = bool(NAME_CUE_RE.search(t))
    toks = [tok for tok in t.split() if tok]
    pure_short = 1 <= len(toks) <= 3 and all(re.fullmatch(r"[A-Za-z][A-Za-z\-']*", tok) for tok in toks)
    if not (has_cue or pure_short):
        return None

    m = NAME_CUE_RE.search(t)
    raw = (m.group(2) if m else t).strip()
    nm = _normalize_person_name(raw)
    if not _looks_like_person_name(nm):
        return None

    st = get_state(session_id) or {}
    if st.get("name") != nm:
        set_state(session_id, name=nm)
        # persist stage/name without changing stage progression
        mark_stage(session_id, stage=(st.get("lead_stage") or "name"), name=nm, source="chat")
        return nm
    return None


# ===== State machine (hints only) =====

def in_progress(session_id: str) -> bool:
    st = get_state(session_id)
    stage = (st or {}).get("lead_stage")
    return bool(stage and stage != "done")


def start(session_id: str, kind: str = "callback") -> Dict[str, str]:
    st = get_state(session_id) or {}
    if st.get("lead_stage") == "done":
        return {"hint": "after_done"}
    set_state(session_id, lead_stage="name", lead_started_at=datetime_now_iso())
    mark_stage(session_id, stage="name", source="chat")
    # first ask is name
    mark_asked(session_id, "name")
    return {"hint": "ask_name"}


def datetime_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _has_contact(st: dict) -> bool:
    return bool((st or {}).get("phone") or (st or {}).get("email"))

def take_turn(session_id: str, text: str) -> dict:
    """
    Pure signals:
      {"hint":"ask_name" | "ask_contact" | "ask_time" | "ask_notes" | "confirm_done" |
       "bridge_back_to_contact" | "bridge_back_to_time"}
    """
    st = get_state(session_id) or {}
    stage = st.get("lead_stage") or "name"

    if stage not in {"name", "contact", "time", "notes", "done"}:
        stage = "name"
        set_state(session_id, lead_stage="name")

    # --- NAME ---
    if stage == "name":
        n = harvest_name(text) or st.get("name")
        if not n:
            return {"hint": "ask_name"}
        set_state(session_id, name=n, lead_stage="contact")
        mark_stage(session_id, stage="contact", name=n, source="chat")
        return {"hint": "ask_contact"}

    # --- CONTACT ---
    if stage == "contact":
        # allow name corrections when safe (your tightened helper)
        _maybe_update_name_from(text, session_id)

        em = harvest_email(text)
        ph = harvest_phone(text)
        if em: set_state(session_id, email=em)
        if ph: set_state(session_id, phone=ph)

        st_now = get_state(session_id) or {}
        if not _has_contact(st_now):
            return {"hint": "ask_contact"}

        set_state(session_id, lead_stage="time")
        mark_stage(session_id, stage="time",
                   email=st_now.get("email"), phone=st_now.get("phone"))
        return {"hint": "ask_time"}

    # --- TIME ---
    if stage == "time":
        # Guard: never progress if we still don't have contact
        if not _has_contact(st):
            return {"hint": "bridge_back_to_contact"}

        pref = (text or "").strip()
        if not pref:
            return {"hint": "ask_time"}

        set_state(session_id, preferred_time=pref, lead_stage="notes")
        mark_stage(session_id, stage="notes", preferred_time=pref)
        return {"hint": "ask_notes"}

    # --- NOTES ---
    if stage == "notes":
        notes = (text or "").strip()
        if notes:
            set_state(session_id, notes=notes)

        st_now = get_state(session_id) or {}
        # Safety gates: only finalize when we truly have name + contact + time
        if not st_now.get("name"):
            set_state(session_id, lead_stage="name")
            return {"hint": "ask_name"}
        if not _has_contact(st_now):
            set_state(session_id, lead_stage="contact")
            return {"hint": "ask_contact"}
        if not st_now.get("preferred_time"):
            set_state(session_id, lead_stage="time")
            return {"hint": "ask_time"}

        now_iso = _now_iso()
        mark_done(
            session_id,
            name=st_now.get("name"),
            phone=st_now.get("phone"),
            email=st_now.get("email"),
            preferred_time=st_now.get("preferred_time"),
            notes=st_now.get("notes") or notes or None,
            done_at=now_iso,
        )
        set_state(session_id, lead_stage="done", lead_done_at=now_iso, lead_just_done=True)
        return {"hint": "confirm_done"}

    # --- DONE ---
    return {"hint": "confirm_done"}