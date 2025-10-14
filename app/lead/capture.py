# app/lead/capture.py
from __future__ import annotations

import os
import re
import json
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

from app.core.session_mem import (
    get_state,
    set_state,
    mark_asked,
    recently_asked,
    update_summary,
)
from app.retrieval.leads import mark_stage, mark_done

# ===== LLM fallback config (for name only; conservative) =====
_OPENAI_MODEL = os.getenv("OPENAI_EXTRACT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_NAME_LLM_MIN_CONF = float(os.getenv("NAME_LLM_MIN_CONF", "0.92"))

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
NAME_RE   = re.compile(
    r"\b(i'?m|i am|this is|my name is|name is|name\s*:|it'?s)\s+([A-Za-z][A-Za-z\-\' ]{1,40})",
    re.I,
)

# Time heuristics: very light, safe to accept as a user preference text
DAY_WORDS = r"(mon|tue|wed|thu|thur|thurs|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today)"
TIME_RE = re.compile(
    rf"\b({DAY_WORDS}(?:\s+at|\s+around|\s+by)?\s*\d{{1,2}}(?::\d{{2}})?\s*(?:am|pm)?)\b"
    r"|(\bany ?time\b|\bmorning\b|\bafternoon\b|\bevening\b|\b3-5 ?pm\b|\b\d{1,2}\s*-\s*\d{1,2}\s*(?:am|pm)\b)",
    re.I,
)

COMPANY_RE = re.compile(
    r"\b(?:my\s+(?:company|business|store|shop)\s+(?:is|called|name(?:d)?\s+is)\s+)([A-Za-z0-9&\-' ]{2,64})",
    re.I
)

_STOP_WORDS = {
    "hi","hello","hey","ok","okay","thanks","thank","please",
    "email","phone","number","call","start","begin","book","callme",
    "price","pricing","cost","whatsapp","chatbot","bot","website","address",
    "yes","yep","yeah","no","nope","what","why","how","when","where","who","can","could",
    "would","does","do","help","service",
}

def harvest_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text or "")
    return m.group(0) if m else None

def harvest_phone(text: str) -> Optional[str]:
    m = PHONE_RE.search(text or "")
    return m.group(0).strip() if m else None

def harvest_company(text: str) -> Optional[str]:
    m = COMPANY_RE.search(text or "")
    if m:
        return " ".join(m.group(1).split()).strip()
    return None

def harvest_time(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    m = TIME_RE.search(t)
    if m:
        val = (m.group(1) or m.group(2) or "").strip()
        return re.sub(r"\s+", " ", val).lower()
    # simple “tuesday 11am?” split forms
    if re.search(DAY_WORDS, t, re.I) and re.search(r"\b\d{1,2}\s*(?:am|pm)\b", t, re.I):
        return re.sub(r"\s+", " ", t).lower()
    # catch “anytime” etc
    if re.search(r"\bany ?time\b|\bmorning\b|\bafternoon\b|\bevening\b", t, re.I):
        return re.sub(r"\s+", " ", t).lower()
    return None

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
        "If none, use null + confidence 0."
    )
    user = f"User message:\n{text or ''}\n"
    try:
        resp = cli.chat.completions.create(
            model=_OPENAI_MODEL,
            temperature=0.0,
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":user},
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
    m = NAME_RE.search(t)
    if m:
        raw = m.group(2).strip()
        nm = _normalize_person_name(raw)
        if _looks_like_person_name(nm):
            return nm
    tokens = [tok for tok in t.split() if tok]
    if 1 <= len(tokens) <= 4 and all(re.fullmatch(r"[A-Za-z][A-Za-z\-']*", tok) for tok in tokens):
        nm = _normalize_person_name(t)
        if _looks_like_person_name(nm):
            return nm
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

def start(session_id: str, kind: str = "callback") -> Dict[str, str]:
    st = get_state(session_id) or {}
    if st.get("lead_stage") == "done":
        return {"hint":"after_done"}
    set_state(session_id, lead_stage="name", lead_started_at=_now_iso())
    mark_stage(session_id, stage="name", source=kind)
    # first ask (with cooldown guard)
    if not recently_asked(session_id, "name", cooldown_sec=45):
        mark_asked(session_id, "name")
        return {"hint":"ask_name"}
    return {"hint":"bridge_back_to_name"}

def _maybe_update_name_from(text: str, session_id: str) -> Optional[str]:
    t = (text or "").strip()
    if not t or _contains_contactish(t):
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
        update_summary(session_id)
        return new_name
    return None

def _maybe_update_company_from(text: str, session_id: str) -> Optional[str]:
    comp = harvest_company(text or "")
    if comp:
        st = get_state(session_id) or {}
        if (st.get("company") or "").strip().lower() != comp.lower():
            set_state(session_id, company=comp)
            update_summary(session_id)
            return comp
    return None

def _ensure_contact_in_state(session_id: str, text: str) -> None:
    em = harvest_email(text)
    ph = harvest_phone(text)
    updates = {}
    if em: updates["email"] = em
    if ph: updates["phone"] = ph
    if updates:
        set_state(session_id, **updates)
        mark_stage(session_id, stage=get_state(session_id).get("lead_stage") or "contact", **updates)
        update_summary(session_id)

def _ensure_time_in_state(session_id: str, text: str) -> None:
    pref = harvest_time(text or "")
    if pref:
        set_state(session_id, preferred_time=pref)
        mark_stage(session_id, stage=get_state(session_id).get("lead_stage") or "time", preferred_time=pref)
        update_summary(session_id)

def _build_lead_report(st: Dict[str, Any]) -> Dict[str, Any]:
    # very small inference; keep it robust
    interest = "high" if st.get("phone") or st.get("email") else "medium"
    quality  = 2 if interest == "high" else 1
    topic = st.get("current_topic") or "not_set"
    company = st.get("company") or "not_set"
    notes = st.get("notes") or None
    when = st.get("preferred_time") or "unspecified"
    name = st.get("name") or "unknown"

    return {
        "summary": f"{name} ({company}) asked about {topic}; prefers time: {when}.",
        "interest_level": interest,
        "lead_quality": quality,            # 0..3
        "next_action": "schedule_callback" if (st.get("phone") or st.get("email")) else "nurture_email",
        "channel": "chat",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": {
            "name": st.get("name"),
            "company": company,
            "phone": st.get("phone"),
            "email": st.get("email"),
            "preferred_time": st.get("preferred_time"),
            "topic": topic,
            "user_notes": notes,
        },
    }

# ===== Main controller =====

def take_turn(session_id: str, text: str) -> Dict[str, str]:
    """
    State machine (signals only).
    Stages: name -> contact -> time -> notes -> done
    Minimal finalize: require name AND (phone or email). Time optional.
    """
    st = get_state(session_id) or {}
    stage = st.get("lead_stage") or "name"

    # opportunistic harvests every turn
    _maybe_update_company_from(text, session_id)
    _ensure_contact_in_state(session_id, text)
    _ensure_time_in_state(session_id, text)
    _maybe_update_name_from(text, session_id)  # safe correction

    # normalize stage
    if stage not in {"name", "contact", "time", "notes", "done"}:
        stage = "name"
        set_state(session_id, lead_stage="name")

    # --- NAME ---
    if stage == "name":
        nm = harvest_name(text) or st.get("name")
        if nm:
            set_state(session_id, name=nm, lead_stage="contact")
            mark_stage(session_id, stage="contact", name=nm, source="chat")
            update_summary(session_id)
            if not recently_asked(session_id, "contact", 45):
                mark_asked(session_id, "contact")
                return {"hint":"ask_contact"}
            return {"hint":"bridge_back_to_contact"}
        # ask (or bridge) name
        if not recently_asked(session_id, "name", 45):
            mark_asked(session_id, "name")
            return {"hint":"ask_name"}
        return {"hint":"bridge_back_to_name"}

    # --- CONTACT ---
    if stage == "contact":
        st_now = get_state(session_id) or {}
        if st_now.get("phone") or st_now.get("email"):
            set_state(session_id, lead_stage="time")
            mark_stage(session_id, stage="time")
            update_summary(session_id)
            # If time already present, advance to notes straight away.
            st_now2 = get_state(session_id) or {}
            if st_now2.get("preferred_time"):
                set_state(session_id, lead_stage="notes")
                mark_stage(session_id, stage="notes")
                update_summary(session_id)
                if not recently_asked(session_id, "notes", 45):
                    mark_asked(session_id, "notes")
                    return {"hint":"ask_notes"}
                return {"hint":"bridge_back_to_notes"}
            # else ask/bridge time
            if not recently_asked(session_id, "time", 45):
                mark_asked(session_id, "time")
                return {"hint":"ask_time"}
            return {"hint":"bridge_back_to_time"}

        # need contact
        if not recently_asked(session_id, "contact", 45):
            mark_asked(session_id, "contact")
            return {"hint":"ask_contact"}
        return {"hint":"bridge_back_to_contact"}

    # --- TIME (optional, but nice to have) ---
    if stage == "time":
        st_now = get_state(session_id) or {}
        pref = st_now.get("preferred_time") or harvest_time(text or "")
        if pref:
            set_state(session_id, preferred_time=pref, lead_stage="notes")
            mark_stage(session_id, stage="notes", preferred_time=pref)
            update_summary(session_id)
            if not recently_asked(session_id, "notes", 45):
                mark_asked(session_id, "notes")
                return {"hint":"ask_notes"}
            return {"hint":"bridge_back_to_notes"}
        # ask/bridge time again with cooldown
        if not recently_asked(session_id, "time", 45):
            mark_asked(session_id, "time")
            return {"hint":"ask_time"}
        return {"hint":"bridge_back_to_time"}

    # --- NOTES + FINALIZE ---
    if stage == "notes":
        # persist notes if any
        user_notes = (text or "").strip()
        if user_notes:
            set_state(session_id, notes=user_notes)
            update_summary(session_id)

        # gate: require name AND (phone or email)
        st_now = get_state(session_id) or {}
        has_name = bool(st_now.get("name"))
        has_contact = bool(st_now.get("phone") or st_now.get("email"))

        if not has_name:
            return {"hint":"bridge_back_to_name"}
        if not has_contact:
            return {"hint":"bridge_back_to_contact"}

        # time optional; proceed
        report = _build_lead_report(st_now)
        # merge report into notes (stored as JSON text)
        merged_notes = report
        if user_notes:
            merged_notes["details"]["user_notes"] = user_notes

        now_iso = _now_iso()
        try:
            mark_stage(
                session_id,
                stage="done",
                name=st_now.get("name"),
                phone=st_now.get("phone"),
                email=st_now.get("email"),
                preferred_time=st_now.get("preferred_time"),
                notes=json.dumps(merged_notes),
            )
            mark_done(
                session_id,
                name=st_now.get("name"),
                phone=st_now.get("phone"),
                email=st_now.get("email"),
                preferred_time=st_now.get("preferred_time"),
                notes=json.dumps(merged_notes),
                done_at=now_iso,
            )
            set_state(session_id, lead_stage="done", lead_done_at=now_iso, lead_just_done=True)
            update_summary(session_id)
            return {"hint":"confirm_done"}
        except Exception:
            # If DB upsert fails, do not crash the API; ask to confirm contact again.
            return {"hint":"bridge_back_to_contact"}

    # --- DONE ---
    if stage == "done":
        return {"hint":"after_done"}

    # Fallback
    return {"hint":"ask_name"}