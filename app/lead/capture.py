# app/lead/capture.py
from __future__ import annotations

import os
import re
import json
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict

from app.core.config import ASK_COOLDOWN_SEC
from app.core.session_mem import (
    get_state,
    set_state,
    mark_asked,
    recently_asked,
)
from app.retrieval.leads import mark_stage, mark_done

"""
This module is the LEAD FLOW CONTROLLER (logic owner).
- It never emits user-facing copy. It returns compact HINT SIGNALS that the generator will phrase.
- Signals:
    {"hint":"ask_name"} | {"hint":"ask_contact"} | {"hint":"ask_time"} | {"hint":"ask_notes"}
    {"hint":"bridge_back_to_name"} | {"hint":"bridge_back_to_contact"} | {"hint":"bridge_back_to_time"}
    {"hint":"confirm_done"} | {"hint":"after_done"}
- Stages: name -> contact -> time -> notes -> done
- Persist on every transition via mark_stage/mark_done.
- Use mark_asked()/recently_asked() to avoid repeating the same ask within a cooldown.
"""

# ===== LLM fallback config (cheap + safe) =====
_OPENAI_MODEL = os.getenv("OPENAI_EXTRACT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
# Be conservative to avoid false positives
_NAME_LLM_MIN_CONF = float(os.getenv("NAME_LLM_MIN_CONF", "0.92"))

_client = None
def _get_client():
    """Lazy import so the app still runs without OpenAI configured."""
    global _client
    if _client is None:
        try:
            from openai import OpenAI  # type: ignore
            _client = OpenAI()
        except Exception:
            _client = False
    return _client

# ===== Fast extractors =====
EMAIL_RE  = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE  = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")

# Broadened name cues: I'm / I am / this is / my name is / name is / name: / it's
NAME_RE   = re.compile(
    r"\b(i'?m|i am|this is|my name is|name is|name\s*:|it'?s)\s+([A-Za-z][A-Za-z\-\' ]{1,40})",
    re.I,
)

# Expanded stop words to block interrogatives and generic phrases from looking like names
_STOP_WORDS = {
    "hi","hello","hey","ok","okay","thanks","thank","please",
    "email","phone","number","call","start","begin","book","callme",
    "price","pricing","cost","whatsapp","chatbot","bot","website","address",
    "yes","yep","yeah","no","nope",
    # interrogatives / generic helpers
    "what","why","how","when","where","who","can","could","would","does","do","help","service",
}

# ----- public (used by main.py for opportunistic harvest) -----

def harvest_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text or "")
    return m.group(0) if m else None

def harvest_phone(text: str) -> Optional[str]:
    m = PHONE_RE.search(text or "")
    return m.group(0).strip() if m else None

def harvest_name(text: str) -> Optional[str]:
    """
    Hybrid extraction:
      1) Broadened regex (cued names)
      2) Plain-name heuristic (entire message looks like a short name)
      3) LLM fallback with confidence + validation
    """
    t = (text or "").strip()

    # 1) Regex with cues
    m = NAME_RE.search(t)
    if m:
        raw = m.group(2).strip()
        nm = _normalize_person_name(raw)
        if _looks_like_person_name(nm):
            return nm

    # 2) Plain-name heuristic (only letters, 1–4 tokens), and not contact-ish
    tokens = [tok for tok in t.split() if tok]
    if 1 <= len(tokens) <= 4 and all(re.fullmatch(r"[A-Za-z][A-Za-z\-']*", tok) for tok in tokens):
        nm = _normalize_person_name(t)
        if _looks_like_person_name(nm):
            return nm

    # 3) LLM fallback
    nm, conf = _llm_extract_name(t)
    if nm and conf >= _NAME_LLM_MIN_CONF:
        nm = _normalize_person_name(nm)
        if _looks_like_person_name(nm):
            return nm

    return None

# ----- internals -----

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _normalize_person_name(raw: str) -> str:
    raw = (raw or "").strip()
    parts = re.split(r"\s+", raw)
    fixed = []
    for p in parts:
        if not p:
            continue
        fixed.append(p[:1].upper() + p[1:].lower())
    return " ".join(fixed)

def _starts_with_interrogative(s: str) -> bool:
    s = (s or "").lstrip().lower()
    for w in ("what","why","how","when","where","who","can","could","would","does","do","please","thanks"):
        if s.startswith(w + " "):
            return True
    return False

def _contains_contactish(s: str) -> bool:
    """Reject strings that look like contact/time chatter rather than a name."""
    sl = (s or "").lower()
    if any(tok in sl for tok in ("number", "phone", "email", "call", "time", "am", "pm", "@", "http", "https", "whatsapp")):
        return True
    if any(ch.isdigit() for ch in sl):
        return True
    return False

def _looks_like_person_name(s: str) -> bool:
    """Heuristic to decide if a value plausibly looks like a person name."""
    if not s:
        return False
    s = s.strip()
    if len(s) < 2 or len(s) > 40:
        return False
    # Hard blocks first
    if _contains_contactish(s):
        return False
    if _starts_with_interrogative(s):
        return False

    tokens = [t for t in re.split(r"\s+", s) if t]
    if not (1 <= len(tokens) <= 4):
        return False

    # If most tokens are stop words, it's not a person name
    sw = sum(1 for t in tokens if t.lower() in _STOP_WORDS)
    if sw >= max(1, len(tokens) - 1):
        return False

    # All tokens should look like name-ish words
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z\-']*", t) for t in tokens):
        return False
    return True

def _llm_extract_name(text: str) -> Tuple[Optional[str], float]:
    """
    Tiny LLM pass: extract a single PERSON name.
    Returns (name, confidence). name=None if none.
    """
    cli = _get_client()
    if not cli:
        return None, 0.0

    system = (
        "Extract exactly one PERSON's name from the user's latest message. "
        "Return compact JSON: {\"name\": string|null, \"confidence\": number}. "
        "If there is no clear person name, use null and confidence 0. "
        "Ignore company/product/brand names, emails, handles, URLs, and phone numbers."
    )
    user = f"User message:\n{text or ''}\n"
    try:
        resp = cli.chat.completions.create(
            model=_OPENAI_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        out = (resp.choices[0].message.content or "").strip()
        data = None
        try:
            data = json.loads(out)
        except Exception:
            m = re.search(r"\{.*\}", out, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
        if isinstance(data, dict):
            nm = (data.get("name") or "").strip() or None
            conf = float(data.get("confidence") or 0.0)
            return nm, conf
    except Exception:
        pass
    return None, 0.0

# Only allow name corrections with a cue or a pure short name;
# never on messages that include contact-ish/time hints.
def _maybe_update_name_from(text: str, session_id: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    if _contains_contactish(t):
        return None

    # Allow if message has a name cue, OR is a pure short name
    has_cue = bool(NAME_RE.search(t))
    tokens = [tok for tok in t.split() if tok]
    pure_short = 1 <= len(tokens) <= 3 and all(re.fullmatch(r"[A-Za-z][A-Za-z\-']*", tok) for tok in tokens)

    if not (has_cue or pure_short):
        return None

    new_name = harvest_name(t)
    if not new_name:
        return None

    st = get_state(session_id) or {}
    if st.get("name") != new_name:
        set_state(session_id, name=new_name)
        # Persist name immediately at whatever stage we’re in
        mark_stage(session_id, stage=(st.get("lead_stage") or "name"), name=new_name)
        return new_name
    return None

# ===== controller API =====

def in_progress(session_id: str) -> bool:
    st = get_state(session_id)
    stage = (st or {}).get("lead_stage")
    return bool(stage and stage != "done")

def start(session_id: str, kind: str = "callback") -> Dict[str, str]:
    """
    Begin a lead capture flow. Stage -> 'name'
    Also persists an initial row with stage='name'.
    Returns a hint dict (no user-facing copy).
    """
    st = get_state(session_id) or {}
    if st.get("lead_stage") == "done":
        # Flow already completed for this session; do not restart.
        return {"hint": "after_done"}

    # Initialize flow
    set_state(session_id, lead_stage="name", lead_kind=kind, lead_started_at=_now_iso())
    mark_stage(session_id, stage="name", source="chat")
    # one-ask-per-turn: stamp the ask when we first start
    mark_asked(session_id, "name")
    return {"hint": "ask_name"}

def take_turn(session_id: str, text: str) -> Dict[str, str]:
    """
    Advance the state machine one step and persist at each transition.
    Returns a hint dict for the LLM to phrase.
    Stages: name -> contact -> time -> notes -> done
    """
    st = get_state(session_id) or {}
    stage = st.get("lead_stage") or "name"

    if stage not in {"name", "contact", "time", "notes", "done"}:
        stage = "name"
        set_state(session_id, lead_stage="name")
        mark_stage(session_id, stage="name", source="chat")
        mark_asked(session_id, "name")
        return {"hint": "ask_name"}

    # --- NAME ---
    if stage == "name":
        n = harvest_name(text) or st.get("name")
        if not n:
            if recently_asked(session_id, "name", ASK_COOLDOWN_SEC):
                return {"hint": "bridge_back_to_name"}
            mark_asked(session_id, "name")
            return {"hint": "ask_name"}

        # Accept and move to contact
        set_state(session_id, name=n, lead_stage="contact")
        mark_stage(session_id, stage="contact", name=n, source="chat")
        mark_asked(session_id, "contact")
        return {"hint": "ask_contact"}

    # --- CONTACT ---
    if stage == "contact":
        # Allow “actually it’s Alex” corrections without resetting (only if the message looks like a name)
        _maybe_update_name_from(text, session_id)

        em = harvest_email(text)
        ph = harvest_phone(text)
        if em:
            set_state(session_id, email=em)
        if ph:
            set_state(session_id, phone=ph)

        # Re-read after writes
        st_now = get_state(session_id) or {}
        if not (st_now.get("email") or st_now.get("phone")):
            if recently_asked(session_id, "contact", ASK_COOLDOWN_SEC):
                return {"hint": "bridge_back_to_contact"}
            mark_asked(session_id, "contact")
            return {"hint": "ask_contact"}

        # Got contact → ask time
        set_state(session_id, lead_stage="time")
        mark_stage(session_id, stage="time", email=st_now.get("email"), phone=st_now.get("phone"))
        mark_asked(session_id, "time")
        return {"hint": "ask_time"}

    # --- TIME ---
    if stage == "time":
        pref = (text or "").strip()
        if not pref:
            if recently_asked(session_id, "time", ASK_COOLDOWN_SEC):
                return {"hint": "bridge_back_to_time"}
            mark_asked(session_id, "time")
            return {"hint": "ask_time"}

        set_state(session_id, preferred_time=pref, lead_stage="notes")
        mark_stage(session_id, stage="notes", preferred_time=pref)
        mark_asked(session_id, "notes")
        return {"hint": "ask_notes"}

    # --- NOTES ---
    if stage == "notes":
        notes = (text or "").strip()
        if notes:
            set_state(session_id, notes=notes)

        st_final = get_state(session_id) or {}
        now_iso = _now_iso()
        mark_done(
            session_id,
            name=st_final.get("name"),
            phone=st_final.get("phone"),
            email=st_final.get("email"),
            preferred_time=st_final.get("preferred_time"),
            notes=st_final.get("notes") or notes or None,
            done_at=now_iso,
        )
        # done + one-shot close flag for UI
        set_state(session_id, lead_stage="done", lead_done_at=now_iso, lead_just_done=True)
        return {"hint": "confirm_done"}

    # --- DONE ---
    return {"hint": "after_done"}