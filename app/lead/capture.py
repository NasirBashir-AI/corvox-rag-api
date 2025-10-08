# app/lead/capture.py
from __future__ import annotations

import os
import re
import json
from datetime import datetime, timezone
from typing import Optional, Tuple

from app.core.session_mem import get_state, set_state
from app.retrieval.leads import mark_stage, mark_done

# ===== LLM fallback config (cheap + safe) =====
_OPENAI_MODEL = os.getenv("OPENAI_EXTRACT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
# Accept only when we’re confident and it passes our validators
_NAME_LLM_MIN_CONF = float(os.getenv("NAME_LLM_MIN_CONF", "0.85"))

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

# Broadened name cues: I'm / I am / this is / my (full) name is / name is / name: / it's
NAME_RE   = re.compile(
    r"\b(i'?m|i am|this is|my (?:full\s+)?name is|name is|name\s*:|it'?s)\s+([A-Za-z][A-Za-z\-\' ]{1,40})",
    re.I,
)

_STOP_WORDS = {
    "hi","hello","hey","ok","okay","thanks","thank","please",
    "email","phone","number","call","start","begin","book","callme",
    "price","pricing","cost","whatsapp","chatbot","bot","website","address",
    "yes","yep","yeah","no","nope"
}

def harvest_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text or "")
    return m.group(0) if m else None

def harvest_phone(text: str) -> Optional[str]:
    m = PHONE_RE.search(text or "")
    return m.group(0).strip() if m else None

def _normalize_person_name(raw: str) -> str:
    raw = (raw or "").strip()
    parts = re.split(r"\s+", raw)
    fixed = []
    for p in parts:
        if not p:
            continue
        fixed.append(p[:1].upper() + p[1:].lower())
    return " ".join(fixed)

def _looks_like_person_name(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    if len(s) < 2 or len(s) > 40:
        return False
    if any(ch.isdigit() for ch in s):
        return False
    if "http" in s.lower() or "@" in s or "_" in s:
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
    """
    Tiny LLM pass: extract a single PERSON name.
    Returns (name, confidence). name=None if none.
    """
    cli = _get_client()
    if not cli:
        return None, 0.0

    system = (
        "Extract exactly one PERSON's name from the user's latest message. "
        'Return compact JSON: {"name": string|null, "confidence": number}. '
        "If there is no clear person name, use null and confidence 0. "
        "Ignore company/product/brand names, emails, handles, URLs, phone numbers."
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
    """
    Hybrid extraction:
      1) Broadened regex
      2) Plain-name heuristic (entire message looks like a name, tolerant to punctuation)
      3) LLM fallback with confidence + validation
    """
    t = (text or "").strip()

    # 1) Regex (leading cues)
    m = NAME_RE.search(t)
    if m:
        raw = m.group(2).strip()
        nm = _normalize_person_name(raw)
        if _looks_like_person_name(nm):
            return nm

    # 2) Plain-name heuristic — tolerant to surrounding words/punctuation.
    #    Try the final chunk of letters (people often end with “…, Nasir”).
    words = re.findall(r"[A-Za-z][A-Za-z\-']*", t)
    if 1 <= len(words) <= 4:
        candidate = _normalize_person_name(" ".join(words))
        if _looks_like_person_name(candidate):
            return candidate
    if words:
        tail = _normalize_person_name(words[-1])
        if _looks_like_person_name(tail):
            return tail

    # 3) LLM fallback
    nm, conf = _llm_extract_name(t)
    if nm and conf >= _NAME_LLM_MIN_CONF:
        nm = _normalize_person_name(nm)
        if _looks_like_person_name(nm):
            return nm

    return None

# ===== Helpers =====
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
    """
    st = get_state(session_id) or {}
    if st.get("lead_stage") == "done":
        return "We’ve got your details noted. Anything else I can help with?"

    set_state(session_id, lead_stage="name", lead_kind=kind, lead_started_at=_now_iso())
    mark_stage(session_id, stage="name", source="chat")
    return "Great — I can arrange that. What’s your name?"

# Allow a name correction later without derailing the flow
def _maybe_update_name_from(text: str, session_id: str) -> Optional[str]:
    new_name = harvest_name(text)
    if not new_name:
        return None
    st = get_state(session_id) or {}
    if st.get("name") != new_name:
        set_state(session_id, name=new_name)
        # Persist name immediately at whatever stage we’re in
        mark_stage(session_id, stage=(st.get("lead_stage") or "name"), name=new_name)
        return new_name
    return None

def take_turn(session_id: str, text: str) -> str:
    """
    Advance the state machine one step and persist at each transition.
    Returns the next question/confirmation to show the user.
    Stages: name -> contact -> time -> notes -> done
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
            return "Could you share your name (e.g., “I’m Sam Patel”)?"
        set_state(session_id, name=n, lead_stage="contact")
        mark_stage(session_id, stage="contact", name=n, source="chat")
        return f"Thanks, {n}. What’s the best phone number or email to reach you?"

    # --- CONTACT ---
    if stage == "contact":
        # Allow “actually it’s Alex” corrections without resetting
        corrected = _maybe_update_name_from(text, session_id)

        em = harvest_email(text)
        ph = harvest_phone(text)
        if em:
            set_state(session_id, email=em)
        if ph:
            set_state(session_id, phone=ph)

        # Re-read minimal state snapshot after writes
        st_now = get_state(session_id) or {}
        if not (st_now.get("email") or st_now.get("phone")):
            if corrected:
                return f"Noted, {corrected}. Could you share a phone number or email?"
            return "I didn’t catch a valid phone or email. Please share one of them."

        set_state(session_id, lead_stage="time")
        mark_stage(
            session_id,
            stage="time",
            email=st_now.get("email"),
            phone=st_now.get("phone"),
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
        return "All set! I’ve logged your request and will close this chat now. We’ll be in touch shortly."

    # --- DONE (or unknown) ---
    set_state(session_id, lead_stage="done")
    return "We’ve got your details noted and I’ll close this chat now. You can start a new chat anytime."