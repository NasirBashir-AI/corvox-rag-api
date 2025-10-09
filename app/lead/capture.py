# app/lead/capture.py
from __future__ import annotations

import os
import re
import json
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict

from app.core.session_mem import (
    get_state,
    set_state,
    mark_asked,
    recently_asked,
)
from app.retrieval.leads import mark_stage, mark_done

# ===== Config =====
_OPENAI_MODEL = os.getenv("OPENAI_EXTRACT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_NAME_LLM_MIN_CONF = float(os.getenv("NAME_LLM_MIN_CONF", "0.92"))
_ASK_COOLDOWN_SEC = int(os.getenv("ASK_COOLDOWN_SEC", "60"))  # don't re-ask same field within this window

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

_STOP_WORDS = {
    "hi","hello","hey","ok","okay","thanks","thank","please",
    "email","phone","number","call","start","begin","book","callme",
    "price","pricing","cost","whatsapp","chatbot","bot","website","address",
    "yes","yep","yeah","no","nope",
    "what","why","how","when","where","who","can","could","would","does","do","help","service",
}

def harvest_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text or "")
    return m.group(0) if m else None

def harvest_phone(text: str) -> Optional[str]:
    m = PHONE_RE.search(text or "")
    return m.group(0).strip() if m else None

# ===== Name helpers =====
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
    sl = (s or "").lower()
    if any(tok in sl for tok in ("number", "phone", "email", "call", "time", "am", "pm", "@", "http", "https", "whatsapp")):
        return True
    if any(ch.isdigit() for ch in sl):
        return True
    return False

def _looks_like_person_name(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    if len(s) < 2 or len(s) > 40:
        return False
    if _contains_contactish(s):
        return False
    if _starts_with_interrogative(s):
        return False

    tokens = [t for t in re.split(r"\s+", s) if t]
    if not (1 <= len(tokens) <= 4):
        return False

    sw = sum(1 for t in tokens if t.lower() in _STOP_WORDS)
    if sw >= max(1, len(tokens) - 1):
        return False

    if not all(re.fullmatch(r"[A-Za-z][A-Za-z\-']*", t) for t in tokens):
        return False
    return True

def _llm_extract_name(text: str) -> Tuple[Optional[str], float]:
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

def harvest_name(text: str) -> Optional[str]:
    t = (text or "").strip()

    # 1) Cued
    m = NAME_RE.search(t)
    if m:
        raw = m.group(2).strip()
        nm = _normalize_person_name(raw)
        if _looks_like_person_name(nm):
            return nm

    # 2) Plain short name
    tokens = [tok for tok in t.split() if tok]
    if 1 <= len(tokens) <= 4 and all(re.fullmatch(r"[A-Za-z][A-Za-z\-']*", tok) for tok in tokens):
        nm = _normalize_person_name(t)
        if _looks_like_person_name(nm):
            return nm

    # 3) LLM fallback (conservative)
    nm, conf = _llm_extract_name(t)
    if nm and conf >= _NAME_LLM_MIN_CONF:
        nm = _normalize_person_name(nm)
        if _looks_like_person_name(nm):
            return nm

    return None

# ===== Flow helpers =====
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def in_progress(session_id: str) -> bool:
    st = get_state(session_id)
    stage = (st or {}).get("lead_stage")
    return bool(stage and stage != "done")

def start(session_id: str, kind: str = "callback") -> str:
    st = get_state(session_id) or {}
    if st.get("lead_stage") == "done":
        return "We’ve got your details noted. Anything else I can help with?"

    set_state(session_id, lead_stage="name", lead_kind=kind, lead_started_at=_now_iso())
    mark_stage(session_id, stage="name", source="chat")
    # We don't stamp asked-for-name here; we do it when we *actually* ask in next_hint.
    return "Great — I can arrange that. What’s your name?"

def _maybe_update_name_from(text: str, session_id: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    if _contains_contactish(t):
        return None

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
        mark_stage(session_id, stage=(st.get("lead_stage") or "name"), name=new_name)
        return new_name
    return None

# ===== Signal engine =====
def next_hint(session_id: str, text: str) -> Dict[str, str]:
    """
    Pure logic (no user-facing copy). Returns a small signal dict:
      {"hint": "..."} where hint in:
        ask_name | bridge_back_to_name
        ask_phone_or_email | bridge_back_to_contact
        ask_time | bridge_back_to_time
        ask_notes | bridge_back_to_notes
        confirm_done
    """
    st = get_state(session_id) or {}
    stage = st.get("lead_stage") or "name"

    # Normalize unexpected stage
    if stage not in {"name", "contact", "time", "notes", "done"}:
        set_state(session_id, lead_stage="name")
        stage = "name"

    # NAME
    if stage == "name":
        n = harvest_name(text) or st.get("name")
        if not n:
            # ask, with cooldown
            if not recently_asked(session_id, "name", _ASK_COOLDOWN_SEC):
                mark_asked(session_id, "name")
                return {"hint": "ask_name"}
            return {"hint": "bridge_back_to_name"}
        # advance
        set_state(session_id, name=n, lead_stage="contact")
        mark_stage(session_id, stage="contact", name=n, source="chat")
        # entering contact -> ask once
        if not recently_asked(session_id, "phone", _ASK_COOLDOWN_SEC) and not recently_asked(session_id, "email", _ASK_COOLDOWN_SEC):
            mark_asked(session_id, "phone")
            mark_asked(session_id, "email")
        return {"hint": "ask_phone_or_email"}

    # CONTACT
    if stage == "contact":
        _maybe_update_name_from(text, session_id)  # optional correction
        em = harvest_email(text)
        ph = harvest_phone(text)
        if em:
            set_state(session_id, email=em)
        if ph:
            set_state(session_id, phone=ph)

        st_now = get_state(session_id) or {}
        if not (st_now.get("email") or st_now.get("phone")):
            # ask/bridge with cooldown
            if not (recently_asked(session_id, "phone", _ASK_COOLDOWN_SEC) or
                    recently_asked(session_id, "email", _ASK_COOLDOWN_SEC)):
                mark_asked(session_id, "phone")
                mark_asked(session_id, "email")
                return {"hint": "ask_phone_or_email"}
            return {"hint": "bridge_back_to_contact"}

        # advance to time
        set_state(session_id, lead_stage="time")
        mark_stage(session_id, stage="time", email=st_now.get("email"), phone=st_now.get("phone"))
        if not recently_asked(session_id, "time", _ASK_COOLDOWN_SEC):
            mark_asked(session_id, "time")
        return {"hint": "ask_time"}

    # TIME
    if stage == "time":
        pref = (text or "").strip()
        if not pref:
            if not recently_asked(session_id, "time", _ASK_COOLDOWN_SEC):
                mark_asked(session_id, "time")
                return {"hint": "ask_time"}
            return {"hint": "bridge_back_to_time"}
        set_state(session_id, preferred_time=pref, lead_stage="notes")
        mark_stage(session_id, stage="notes", preferred_time=pref)
        if not recently_asked(session_id, "notes", _ASK_COOLDOWN_SEC):
            mark_asked(session_id, "notes")
        return {"hint": "ask_notes"}

    # NOTES
    if stage == "notes":
        notes = (text or "").strip()
        if notes:
            set_state(session_id, notes=notes)

        st_now = get_state(session_id) or {}
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

    # DONE
    return {"hint": "confirm_done"}

# ===== Back-compat wrapper (returns short sentence) =====
def take_turn(session_id: str, text: str) -> str:
    """
    Compatibility layer for existing main.py.
    Uses next_hint(...) internally and maps the hint to a short microcopy.
    """
    sig = next_hint(session_id, text)
    hint = (sig or {}).get("hint", "")

    st = get_state(session_id) or {}
    name = st.get("name")

    # Minimal, human phrasing; LLM will still see the hint via context in main.py.
    if hint == "ask_name":
        return "What’s your name?"
    if hint == "bridge_back_to_name":
        return "Before we proceed, may I take your name so we can arrange the callback?"

    if hint == "ask_phone_or_email":
        # implicit confirmation of name without asking a yes/no
        if name:
            return f"Thanks, {name}. What’s the best phone number or email to reach you?"
        return "What’s the best phone number or email to reach you?"
    if hint == "bridge_back_to_contact":
        return "When you’re ready, please share a phone number or email so we can book the call."

    if hint == "ask_time":
        return "When is a good time (and timezone) for us to contact you?"
    if hint == "bridge_back_to_time":
        return "We can work around your schedule — what time suits you (and which timezone)?"

    if hint == "ask_notes":
        return "Got it. Any extra context about your needs? (optional)"
    if hint == "bridge_back_to_notes":
        return "If there’s any extra context you’d like to add, feel free to share it."

    # confirm_done (and any unknown)
    return "All set! I’ve logged your request. We’ll contact you shortly. Anything else I can help with?"