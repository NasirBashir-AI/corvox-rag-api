# app/api/main.py
"""
FastAPI entrypoint for Corah.

Endpoints
- GET  /api/health  : liveness probe
- GET  /api/search  : retrieval-only probe (debug)
- POST /api/chat    : main chat orchestration (Phase 2 ready)
- POST /api/close   : explicit session close (front-end button)

Notes
- No schema refactors required; we import existing pydantic models.
- Adds Phase-2 logic: sentiment+intent scoring, adaptive flow, lead priority,
  recap-before-save, audit note, session_closed flag, inactivity timeout.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import re

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# === Internal imports (match your repo) ===
from app.api.schemas import (
    HealthResponse,               # ok: bool
    SearchHit, SearchResponse,    # hits: List[SearchHit]
    ChatRequest, ChatResponse,    # question, session_id -> reply, session_closed, meta
)
from app.retrieval.retriever import search
from app.generation.generator import generate_reply  # your LLM-first sandwich
from app.generation.lead_intent import classify_lead_intent
from app.api.intents import route_intent   # lightweight router (smalltalk/contact/etc.)
from app.core.session_mem import (
    get_state,           # (session_id) -> dict state or {}
    set_state,           # (session_id, **fields) -> None
    ensure_session,      # (session_id) -> str (returns id)
    touch_session,       # (session_id) -> None   (update last_active)
    summarize_session,   # (session_id) -> str    (short rolling summary)
    clear_session,       # (session_id) -> None
)
from app.lead.capture import (
    harvest_name, harvest_email, harvest_phone,
    build_lead_payload, save_lead,   # returns saved id / dict
)
from app.retrieval.leads import normalize_priority  # if present; else returns as-is
from app.core.config import INACTIVITY_MINUTES, ALLOW_SMALLTALK, CORS_ALLOW_ORIGINS

# ---------------------------
# App
# ---------------------------

app = FastAPI(title="Corah API", version="2.0-phase2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Helpers (Phase-2 local)
# ---------------------------

_POSITIVE = ("great", "thanks", "thank you", "appreciate", "perfect", "awesome", "good", "nice")
_FRUSTRATED = ("not helpful", "annoying", "useless", "waste", "rubbish", "angry", "frustrated", "upset", "stupid", "bloody")
_BUYING_SIGNALS = ("proposal", "quote", "pricing", "how much", "cost", "book", "call me", "phone", "email me", "start", "go ahead")

def _lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _now() -> datetime:
    return datetime.now(timezone.utc)

def _elapsed_minutes(dt: Optional[datetime]) -> float:
    if not dt:
        return 1e9
    delta = _now() - dt
    return delta.total_seconds() / 60.0

def _heuristic_sentiment(text: str) -> str:
    t = _lower(text)
    if any(kw in t for kw in _FRUSTRATED):
        return "frustrated"
    if any(kw in t for kw in _POSITIVE):
        return "positive"
    return "neutral"

def _priority_from_intent(interest_level: str, contact_given: bool) -> str:
    # Simple, clear mapping (Hot/Warm/Cold)
    if interest_level in ("buying", "explicit_cta"):
        return "hot"
    if contact_given:
        return "hot"
    if interest_level == "curious":
        return "warm"
    return "cold"

def _needs_final_check(user_text: str) -> bool:
    """Return True if user said 'thanks' but might still continue (we ask once)."""
    t = _lower(user_text)
    return "thank" in t or "thanks" in t

def _is_goodbye(user_text: str) -> bool:
    t = _lower(user_text)
    return any(x in t for x in ("bye", "that’s all", "thats all", "that is all", "no, that’s all", "no thats all", "no im good", "no i’m good", "no i am good", "no more", "all good now"))

def _contact_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": state.get("name"),
        "email": state.get("email"),
        "phone": state.get("phone"),
        "company": state.get("company"),
        "preferred_time": state.get("preferred_time"),
    }

def _compose_audit_note(sentiment: str, intent: str, priority: str, signals: List[str], corrections: List[str], closure_type: str) -> str:
    parts = []
    parts.append(f"Sentiment={sentiment}")
    parts.append(f"Intent={intent}")
    parts.append(f"Priority={priority}")
    if signals:
        parts.append("Signals=" + ", ".join(signals[:6]))
    if corrections:
        parts.append("Corrections=" + "; ".join(corrections[:5]))
    parts.append(f"Closure={closure_type}")
    return " | ".join(parts)

# ---------------------------
# Health
# ---------------------------

@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True)

# ---------------------------
# Retrieval probe
# ---------------------------

@app.get("/api/search", response_model=SearchResponse)
def api_search(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=20)) -> SearchResponse:
    try:
        hits_raw = search(q, k=k)
        hits: List[SearchHit] = [
            SearchHit(score=hit.get("score", 0.0), content=hit.get("content", ""), source=hit.get("source"))
            for hit in (hits_raw or [])
        ]
        return SearchResponse(hits=hits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search_failed: {e}")

# ---------------------------
# Explicit close (front-end button)
# ---------------------------

@app.post("/api/close")
def api_close(session_id: Optional[str] = None) -> Dict[str, Any]:
    if not session_id:
        return {"ok": True, "session_closed": True}
    try:
        # Finalize and clear
        st = get_state(session_id) or {}
        summarize_session(session_id)  # refresh short summary
        set_state(session_id, closed_at=_now().isoformat())
        clear_session(session_id)
        return {"ok": True, "session_closed": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"close_failed: {e}")

# ---------------------------
# Main chat endpoint
# ---------------------------

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    """
    Phase-2 orchestration:
      - Ensure session
      - Inactivity handling
      - Lightweight sentiment (heuristic) + lead intent classifier
      - Adaptive close gate (final check on casual 'thanks')
      - Recap-before-save & Audit note on close
      - session_closed flag for the widget
    """
    # Ensure session & touch activity
    sid = ensure_session(req.session_id)
    touch_session(sid)

    # ---- Inactivity: if the client's last_active is very old, neutral-close ----
    st = get_state(sid) or {}
    last_active_at: Optional[datetime] = st.get("last_active_at")  # may be a datetime in your impl
    # If your session store keeps ISO strings, try to parse:
    if isinstance(last_active_at, str):
        try:
            last_active_at = datetime.fromisoformat(last_active_at)
        except Exception:
            last_active_at = None

    if _elapsed_minutes(last_active_at) > (INACTIVITY_MINUTES or 5) + 1e-6:
        # Close due to inactivity, do not pretend a friendly goodbye
        summarize_session(sid)
        audit_note = _compose_audit_note(
            sentiment="neutral",
            intent="none",
            priority="cold",
            signals=[],
            corrections=[],
            closure_type="timeout",
        )
        # save lead only if we already have contact details:
        if st.get("email") or st.get("phone") or st.get("name"):
            payload = build_lead_payload(
                session_id=sid,
                session_summary=st.get("session_summary", ""),
                contact=_contact_snapshot(st),
                metadata={"audit_note": audit_note, "closure_type": "timeout"},
            )
            save_lead(payload)
        clear_session(sid)
        return ChatResponse(
            reply="This chat is now closing due to inactivity. You can start a new one anytime.",
            session_id=sid,
            session_closed=True,
            meta={"closure": "timeout"}
        )

    user_q = req.question or ""
    t = _lower(user_q)

    # ---- Route simple intents (smalltalk etc.) early if enabled ----
    routed = route_intent(user_q)  # may return {"kind":"smalltalk", "reply":"...", ...} or None
    if routed and routed.get("kind") == "smalltalk" and ALLOW_SMALLTALK:
        # still keep sentiment/intent for analytics
        turns = [{"role": "user", "content": user_q}]
        lead_clf = classify_lead_intent(turns) or {}
        # never close on smalltalk; keep it light
        return ChatResponse(
            reply=routed.get("reply", "Hello!"),
            session_id=sid,
            session_closed=False,
            meta={"sentiment": _heuristic_sentiment(user_q),
                  "intent_level": lead_clf.get("interest_level", "none"),
                  "signals": lead_clf.get("signals", [])}
        )

    # ---- Sentiment (heuristic) and Intent (LLM classifier) ----
    sentiment = _heuristic_sentiment(user_q)
    turns = [{"role": "user", "content": user_q}]
    lead_clf = classify_lead_intent(turns) or {}
    interest_level = lead_clf.get("interest_level", "none")
    signals = lead_clf.get("signals", [])
    contact_given = bool(harvest_email(user_q) or harvest_phone(user_q))
    priority = _priority_from_intent(interest_level, contact_given)
    priority = normalize_priority(priority) if callable(globals().get("normalize_priority")) else priority

    # ---- Harvest details opportunistically (in-session only; not saved yet) ----
    name = harvest_name(user_q) or st.get("name")
    email = harvest_email(user_q) or st.get("email")
    phone = harvest_phone(user_q) or st.get("phone")
    # simple company capture
    company = st.get("company")
    if not company:
        # naive: look for "my company X" or "from X"
        m = re.search(r"(?:my\s+company|from)\s+([A-Za-z0-9&\-\._\s]{2,40})", t)
        if m:
            company = m.group(1).strip()

    # Update in-session state (not DB lead yet)
    set_state(
        sid,
        name=name, email=email, phone=phone, company=company,
        last_active_at=_now().isoformat(),
        last_user_text=user_q,
        last_sentiment=sentiment,
        last_interest_level=interest_level,
        last_priority=priority,
    )
    summarize_session(sid)

    # ---- Adaptive close gate: casual "thanks" needs one final check ----
    if _needs_final_check(user_q) and not _is_goodbye(user_q):
        reply = "You’re very welcome! Is there anything else I can help with before we finish?"
        return ChatResponse(reply=reply, session_id=sid, session_closed=False,
                            meta={"sentiment": sentiment, "intent_level": interest_level, "priority": priority})

    # ---- Goodbye / explicit end handling ----
    if _is_goodbye(user_q) or "send proposal" in t or "prepare a proposal" in t or "call me" in t:
        # recap-before-save (use state snapshot)
        st2 = get_state(sid) or {}
        contact = _contact_snapshot(st2)
        # if no contact, ask for it instead of closing
        if not (contact.get("email") or contact.get("phone")):
            ask = "Before I close, could you share your email or phone so we can follow up?"
            return ChatResponse(reply=ask, session_id=sid, session_closed=False,
                                meta={"sentiment": sentiment, "intent_level": interest_level, "priority": priority, "need_contact": True})

        # Confirm recap (client can show a confirmation step in UI if needed)
        recap = (
            f"Just to confirm: name {contact.get('name') or '—'}, "
            f"email {contact.get('email') or '—'}, phone {contact.get('phone') or '—'}, "
            f"company {contact.get('company') or '—'}. Is that correct?"
        )
        # We send the recap; once the user says "yes/ok", front-end will call /api/chat again
        # with their "yes" and we proceed to save & close.
        set_state(sid, pending_recap=True)
        return ChatResponse(
            reply=recap,
            session_id=sid,
            session_closed=False,
            meta={"recap": True, "sentiment": sentiment, "intent_level": interest_level, "priority": priority}
        )

    # ---- If user responds to recap with confirmation, save and close ----
    if st.get("pending_recap"):
        if any(x in t for x in ("yes", "correct", "that’s right", "thats right", "ok thats fine", "okay that is fine", "looks good")):
            st2 = get_state(sid) or {}
            contact = _contact_snapshot(st2)
            # audit note
            audit_note = _compose_audit_note(
                sentiment=st2.get("last_sentiment", sentiment),
                intent=st2.get("last_interest_level", interest_level),
                priority=st2.get("last_priority", priority),
                signals=signals,
                corrections=st2.get("corrections", []),
                closure_type="user_end"
            )
            payload = build_lead_payload(
                session_id=sid,
                session_summary=st2.get("session_summary", ""),
                contact=contact,
                metadata={
                    "audit_note": audit_note,
                    "closure_type": "user_end"
                },
            )
            save_lead(payload)
            clear_session(sid)
            return ChatResponse(
                reply="Thank you! I’m closing this chat now — the session will end in a moment. You can start a new chat anytime.",
                session_id=sid,
                session_closed=True,
                meta={"closure": "user_end"}
            )
        else:
            # User corrected something; capture corrections and re-recap
            # Try to harvest updated fields
            new_name = harvest_name(user_q) or name
            new_email = harvest_email(user_q) or email
            new_phone = harvest_phone(user_q) or phone
            corrections = []
            if new_name and new_name != name:
                corrections.append(f"name: {name} -> {new_name}")
            if new_email and new_email != email:
                corrections.append(f"email: {email} -> {new_email}")
            if new_phone and new_phone != phone:
                corrections.append(f"phone: {phone} -> {new_phone}")

            set_state(sid, name=new_name, email=new_email, phone=new_phone,
                      corrections=(st.get("corrections", []) + corrections))
            contact = _contact_snapshot(get_state(sid) or {})
            recap = (
                f"Updated. Please confirm: name {contact.get('name') or '—'}, "
                f"email {contact.get('email') or '—'}, phone {contact.get('phone') or '—'}, "
                f"company {contact.get('company') or '—'}. Is that correct?"
            )
            return ChatResponse(
                reply=recap,
                session_id=sid,
                session_closed=False,
                meta={"recap": True, "sentiment": sentiment, "intent_level": interest_level, "priority": priority}
            )

    # ---- Normal path: generate an answer with your generator (RAG sandwich) ----
    try:
        reply = generate_reply(
            question=user_q,
            session_id=sid,
            sentiment=sentiment,
            intent_level=interest_level,
            priority=priority,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generation_failed: {e}")

    return ChatResponse(
        reply=reply,
        session_id=sid,
        session_closed=False,
        meta={"sentiment": sentiment, "intent_level": interest_level, "priority": priority}
    )