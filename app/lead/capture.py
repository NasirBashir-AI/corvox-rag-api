# app/lead/capture.py
from __future__ import annotations

import os
import re
import json
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

from app.core.session_mem import (
    get_state, set_state,
    mark_asked, recently_asked,
)
from app.retrieval.leads import mark_stage, mark_done

# ===== Config =====
_OPENAI_MODEL = os.getenv("OPENAI_EXTRACT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_NAME_LLM_MIN_CONF = float(os.getenv("NAME_LLM_MIN_CONF", "0.92"))

# ===== Lazy OpenAI client =====
_client = None
def _get_client():
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
NAME_CUE_RE = re.compile(r"\b(i'?m|i am|this is|my name is|name is|name\s*:|it'?s)\s+([A-Za-z][A-Za-z\-\' ]{1,40})", re.I)

def harvest_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text or "")
    return m.group(0) if m else None

def harvest_phone(text: str) -> Optional[str]:
    m = PHONE_RE.search(text or "")
    return m.group(0).strip() if m else None

def _normalize_name(raw: str) -> str:
    parts = re.split(r"\s+", (raw or "").strip())
    fixed = [p[:1].upper() + p[1:].lower() for p in parts if p]
    return " ".join(fixed)

def _looks_like_person_name(s: str) -> bool:
    if not s: return False
    s = s.strip()
    if len(s) < 2 or len(s) > 40: return False
    tokens = [t for t in re.split(r"\s+", s) if t]
    if not (1 <= len(tokens) <= 4): return False
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z\-']*", t) for t in tokens): return False
    return True

def _llm_extract_name(text: str) -> Tuple[Optional[str], float]:
    cli = _get_client()
    if not cli:
        return None, 0.0
    system = (
        "Extract exactly one PERSON's name from the user's latest message. "
        "Return JSON: {\"name\": string|null, \"confidence\": number}. "
        "If no clear person name, use null and confidence 0. Ignore brands, emails, URLs, phones."
    )
    user = f"User message:\n{text or ''}\n"
    try:
        resp = cli.chat.completions.create(
            model=_OPENAI_MODEL,
            temperature=0.0,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
        )
        out = (resp.choices[0].message.content or "").strip()
        data = None
        try:
            data = json.loads(out)
        except Exception:
            m = re.search(r"\{.*\}", out, re.DOTALL)
            if m: data = json.loads(m.group(0))
        if isinstance(data, dict):
            nm = (data.get("name") or "").strip() or None
            conf = float(data.get("confidence") or 0.0)
            return nm, conf
    except Exception:
        pass
    return None, 0.0

def harvest_name(text: str) -> Optional[str]:
    t = (text or "").strip()

    # 1) Name cue
    m = NAME_CUE_RE.search(t)
    if m:
        raw = m.group(2).strip()
        nm = _normalize_name(raw)
        if _looks_like_person_name(nm):
            return nm

    # 2) Pure short name
    toks = [tok for tok in t.split() if tok]
    if 1 <= len(toks) <= 3 and all(re.fullmatch(r"[A-Za-z][A-Za-z\-']*", tok) for tok in toks):
        nm = _normalize_name(t)
        if _looks_like_person_name(nm):
            return nm

    # 3) LLM fallback (guarded)
    nm, conf = _llm_extract_name(t)
    if nm and conf >= _NAME_LLM_MIN_CONF:
        nm = _normalize_name(nm)
        if _looks_like_person_name(nm):
            return nm
    return None

# ===== Helpers =====
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def in_progress(session_id: str) -> bool:
    st = get_state(session_id) or {}
    stage = st.get("lead_stage")
    return bool(stage and stage in {"name","contact","time","notes"})

def start(session_id: str, kind: str = "callback") -> Dict[str, Any]:
    """
    Begin a lead capture flow. We return a SIGNAL (no copy).
    """
    st = get_state(session_id) or {}
    if st.get("lead_stage") == "done":
        return {"hint": "after_done"}
    set_state(session_id, lead_stage="name", lead_kind=kind, lead_started_at=_now_iso())
    mark_stage(session_id, stage="name", source="chat")
    # mark first ask only when the LLM actually asks; controller just signals
    return {"hint": "ask_name"}

def take_turn(session_id: str, text: str) -> Dict[str, Any]:
    """
    Controller state machine. Emits SIGNALS only:
      {"hint":"ask_name" | "ask_contact" | "ask_time" | "ask_notes" | "confirm_done" |
               "bridge_back_to_name" | "bridge_back_to_contact" | "bridge_back_to_time" | "bridge_back_to_notes" |
               "after_done"}
    """
    st = get_state(session_id) or {}
    stage = st.get("lead_stage") or "name"

    # Normalize stage if corrupted
    if stage not in {"name","contact","time","notes","done"}:
        stage = "name"
        set_state(session_id, lead_stage="name")

    # ===== NAME =====
    if stage == "name":
        nm = harvest_name(text) or st.get("name")
        if nm:
            if nm != st.get("name"):
                set_state(session_id, name=nm)
                mark_stage(session_id, stage="name", name=nm, source="chat")
            # move to contact
            set_state(session_id, lead_stage="contact")
            mark_stage(session_id, stage="contact")
            return {"hint":"ask_contact"}
        # Ask/bridge gating
        if recently_asked(session_id, "name"):
            return {"hint":"bridge_back_to_name"}
        mark_asked(session_id, "name")
        return {"hint":"ask_name"}

    # ===== CONTACT =====
    if stage == "contact":
        em = harvest_email(text)
        ph = harvest_phone(text)
        wrote = False
        if em and em != st.get("email"):
            set_state(session_id, email=em); wrote = True
        if ph and ph != st.get("phone"):
            set_state(session_id, phone=ph); wrote = True
        st = get_state(session_id) or {}
        if st.get("email") or st.get("phone"):
            set_state(session_id, lead_stage="time")
            mark_stage(session_id, stage="time", email=st.get("email"), phone=st.get("phone"))
            return {"hint":"ask_time"}
        # Ask/bridge gating
        if recently_asked(session_id, "contact"):
            return {"hint":"bridge_back_to_contact"}
        mark_asked(session_id, "contact")
        return {"hint":"ask_contact"}

    # ===== TIME =====
    if stage == "time":
        # We accept time that may have been harvested upstream (preferred_time already set)
        pref = (st.get("preferred_time") or "").strip()
        incoming = (text or "").strip()
        # If user supplied something non-empty, trust upstream time-harvester to have stored it;
        # otherwise, if we already have pref, proceed.
        if incoming or pref:
            if incoming and not pref:
                # fallback: persist raw if upstream missed it
                set_state(session_id, preferred_time=incoming)
                pref = incoming
            set_state(session_id, lead_stage="notes")
            mark_stage(session_id, stage="notes", preferred_time=pref or incoming)
            return {"hint":"ask_notes"}
        # Ask/bridge gating
        if recently_asked(session_id, "time"):
            return {"hint":"bridge_back_to_time"}
        mark_asked(session_id, "time")
        return {"hint":"ask_time"}

    # ===== NOTES (defensive finalize) =====
    if stage == "notes":
        # Save notes opportunistically
        notes = (text or "").strip()
        if notes and notes != (st.get("notes") or ""):
            set_state(session_id, notes=notes)

        # Fresh snapshot after writes
        st = get_state(session_id) or {}
        name  = st.get("name")
        phone = st.get("phone")
        email = st.get("email")
        pref  = st.get("preferred_time")

        # Guard: only finish when all required are present
        if not name:
            return {"hint":"bridge_back_to_name"}
        if not (phone or email):
            return {"hint":"bridge_back_to_contact"}
        if not pref:
            return {"hint":"bridge_back_to_time"}

        # Finalize safely
        now_iso = _now_iso()
        try:
            mark_done(
                session_id,
                name=name,
                phone=phone,
                email=email,
                preferred_time=pref,
                notes=st.get("notes"),
                done_at=now_iso,
            )
            set_state(session_id, lead_stage="done", lead_done_at=now_iso, lead_just_done=True)
            return {"hint":"confirm_done"}
        except Exception as e:
            # If persistence hiccups, don’t crash the API—gracefully steer back
            set_state(session_id, lead_error=str(e))
            return {"hint":"bridge_back_to_notes"}

    # ===== DONE =====
    if stage == "done":
        return {"hint":"after_done"}

    # Fallback
    return {"hint":"ask_name"}