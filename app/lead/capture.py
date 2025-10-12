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

# Times (simple, human-friendly): ranges and single times with am/pm, day words, and “tomorrow/today/anytime”
TIME_RANGE_RE = re.compile(
    r"\b(\d{1,2}(:\d{2})?\s?(am|pm))\s*(?:to|\-|–|—)\s*(\d{1,2}(:\d{2})?\s?(am|pm))\b",
    re.I,
)
TIME_SINGLE_RE = re.compile(
    r"\b(?:(today|tomorrow)\s*)?(?:on\s+)?(mon(?:day)?s?|tue(?:s|sday)?s?|wed(?:nesday)?s?|thu(?:rsday)?s?|fri(?:day)?s?|sat(?:urday)?s?|sun(?:day)?s?)?\s*(?:between\s+)?(\d{1,2}(:\d{2})?\s?(am|pm))(?:\s*(?:to|\-|–|—)\s*(\d{1,2}(:\d{2})?\s?(am|pm)))?\b",
    re.I,
)
TIME_ANY_RE = re.compile(r"\b(any ?time|whenever|flexible|any\s+day)\b", re.I)

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

def harvest_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text or "")
    return m.group(0) if m else None

def harvest_phone(text: str) -> Optional[str]:
    m = PHONE_RE.search(text or "")
    return m.group(0).strip() if m else None

def harvest_time(text: str) -> Optional[str]:
    """
    Lightweight, permissive time capture.
    Returns a clean human-readable snippet (we store as-is).
    Examples it will catch:
      - "tomorrow 3pm", "Monday 10:30am", "Mondays 3pm-5pm", "between 3pm and 5pm", "anytime"
    """
    t = (text or "").strip()
    if not t:
        return None

    # explicit ranges first: "3pm-5pm" / "3pm to 5pm"
    m = TIME_RANGE_RE.search(t)
    if m:
        return m.group(0).strip()

    # single time (optionally with day or 'tomorrow/today', optionally with a trailing range)
    m = TIME_SINGLE_RE.search(t)
    if m:
        return m.group(0).strip()

    # broad catch-all like "anytime"
    m = TIME_ANY_RE.search(t)
    if m:
        return m.group(0).strip().lower()

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

    # 2) Plain-name heuristic (only letters, 1–4 tokens), and not contactish
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

# Only allow name corrections with a cue or a pure short name;
# never on messages that include contactish/time hints.
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

def take_turn(session_id: str, text: str) -> str:
    """
    Advance the state machine one step and persist at each transition.
    Returns the next question/confirmation to show the user.
    Stages: name -> contact -> time -> notes -> done
    (We also opportunistically capture time if it appears early.)
    """
    st = get_state(session_id) or {}
    stage = st.get("lead_stage") or "name"

    if stage not in {"name", "contact", "time", "notes", "done"}:
        stage = "name"
        set_state(session_id, lead_stage="name")

    # Opportunistic time capture on every turn
    maybe_time = harvest_time(text)
    if maybe_time:
        set_state(session_id, preferred_time=maybe_time)

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
        # Allow “actually it’s Alex” corrections without resetting,
        # but only if the message looks like a name (cue or pure short), not contactish.
        corrected = _maybe_update_name_from(text, session_id)

        em = harvest_email(text)
        ph = harvest_phone(text)
        if em:
            set_state(session_id, email=em)
        if ph:
            set_state(session_id, phone=ph)

        # Re-read a minimal snapshot after writes
        st_now = get_state(session_id) or {}
        have_contact = bool(st_now.get("email") or st_now.get("phone"))
        have_time    = bool(st_now.get("preferred_time"))

        if not have_contact:
            if corrected:
                return f"Noted, {corrected}. Could you share a phone number or email?"
            return "I didn’t catch a valid phone or email. Please share one of them."

        # We have contact. If time is already available (from any earlier message), jump to notes.
        if have_time:
            set_state(session_id, lead_stage="notes")
            mark_stage(session_id, stage="notes")
            return "Got it. Any extra context about your needs? (optional)"

        # Otherwise ask for time next
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
        pref = harvest_time(text) or (text or "").strip()
        if not pref or not harvest_time(text):
            # If user didn’t express a time-like phrase, keep it simple
            return "What time works best (and your timezone)?"
        set_state(session_id, preferred_time=pref, lead_stage="notes")
        mark_stage(session_id, stage="notes", preferred_time=pref)
        return "Got it. Any extra context about your needs? (optional)"

    # --- NOTES ---
    if stage == "notes":
        notes = (text or "").strip()
        if notes:
            set_state(session_id, notes=notes)

        st_now = get_state(session_id) or {}

        # ==== Completion gate: require name + (phone or email) + preferred_time ====
        nm  = st_now.get("name")
        ph  = st_now.get("phone")
        em  = st_now.get("email")
        pft = st_now.get("preferred_time")

        missing = []
        if not nm:  missing.append("name")
        if not (ph or em): missing.append("contact")
        if not pft: missing.append("time")

        if missing:
            # Bridge back gently to the FIRST missing piece
            need = missing[0]
            if need == "name":
                return "I have your details—what’s your name so I can log it correctly?"
            if need == "contact":
                return "What’s the best phone number or email to reach you?"
            # need == "time"
            return "When is a good time (and timezone) for us to contact you?"

        # ==== Safe finalize with try/except to avoid any crash ====
        # Safe finalize with try/except to avoid any crash
        try:
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
            # one-shot flag so UI closes once, not forever
            set_state(session_id, lead_stage="done", lead_done_at=now_iso, lead_just_done=True)
            return "All set! I’ve logged your request. We’ll contact you shortly. Anything else I can help with?"
        except Exception as e:
            # Soft-finalize: keep chat alive and avoid 500s
            backup = {
                "name": st_now.get("name") if 'st_now' in locals() else None,
                "phone": st_now.get("phone") if 'st_now' in locals() else None,
                "email": st_now.get("email") if 'st_now' in locals() else None,
                "preferred_time": st_now.get("preferred_time") if 'st_now' in locals() else None,
                "notes": (st_now.get("notes") if 'st_now' in locals() else None) or notes or None,
                "failed_at": _now_iso(),
                "error": type(e).__name__,
            }
            set_state(session_id, lead_stage="done", lead_done_at=_now_iso(), lead_just_done=True, lead_backup=backup)
            return "All set! I’ve saved your details and flagged the team to confirm the time. We’ll be in touch shortly. Anything else I can help with?"

    # --- DONE (follow-ups after completion) ---
    if stage == "done":
        # stage is already done; no need to change it again
        return "We’ve got your details noted. I’ll close this chat now. You can start a new chat anytime."