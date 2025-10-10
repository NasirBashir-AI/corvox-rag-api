# app/api/main.py
"""
FastAPI entrypoint for Corah.
- /api/health  : liveness
- /api/search  : retrieval-only probe
- /api/chat    : LLM-first sandwich (planner -> retrieval -> final), controller=logic / LLM=words
"""

from __future__ import annotations

import re
from typing import List, Optional
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SearchHit,
    SearchResponse,
)

# Import config as a module so we can safely default missing tunables
from app.core import config as CFG

from app.retrieval.retriever import search, get_facts
from app.generation.generator import generate_answer
from app.lead.capture import (
    in_progress as lead_in_progress,
    start as lead_start,
    take_turn as lead_turn,
    harvest_email,
    harvest_phone,
    harvest_name,
)
from app.core.session_mem import (
    get_state,
    set_state,
    append_turn,
    recent_turns,
    mark_asked,
    recently_asked,
)

# ---------------------------
# App + CORS
# ---------------------------

app = FastAPI(
    title="Corah API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
def api_search(
    q: str = Query(..., min_length=1),
    k: int = Query(getattr(CFG, "RETRIEVAL_TOP_K", 5), ge=1, le=20),
) -> SearchResponse:
    try:
        hits_raw = search(q, k=k)
        hits: List[SearchHit] = [
            SearchHit(
                document_id=h.get("document_id"),
                chunk_no=h.get("chunk_no"),
                title=h.get("title"),
                source_uri=h.get("source_uri"),
                content=h.get("content"),
                score=float(h.get("score", 0.0)),
            )
            for h in hits_raw
        ]
        return SearchResponse(hits=hits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search_failed: {type(e).__name__}: {e}")

# ---------------------------
# Helpers (orchestration)
# ---------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _cooldown_ok(last_iso: Optional[str], cooldown_sec: int) -> bool:
    if not last_iso:
        return True
    try:
        last = datetime.fromisoformat(last_iso)
    except Exception:
        return True
    return (_now_utc() - last) >= timedelta(seconds=cooldown_sec)

def _compute_last_asked(state: dict) -> Optional[str]:
    # Pick the most recent asked_for_* timestamp, return its field key
    fields = ("name", "contact", "time", "notes")
    latest_field = None
    latest_ts = None
    for f in fields:
        t_iso = state.get(f"asked_for_{f}_at")
        if not t_iso:
            continue
        try:
            ts = datetime.fromisoformat(t_iso)
        except Exception:
            continue
        if latest_ts is None or ts > latest_ts:
            latest_ts = ts
            latest_field = f
    return latest_field

def _decide_hint_from_stage(state: dict) -> Optional[str]:
    """
    Map current lead_stage + known values to a compact hint the LLM can phrase.
    """
    stage = state.get("lead_stage") or None
    if stage == "done":
        return "confirm_done"

    # Ask mapping by stage
    if stage == "name":
        return "ask_name"

    if stage == "contact":
        # if either phone or email already present, controller should have advanced
        # but if not, we need contact
        has_phone = bool(state.get("phone"))
        has_email = bool(state.get("email"))
        return None if (has_phone or has_email) else "ask_contact"

    if stage == "time":
        return "ask_time" if not state.get("preferred_time") else None

    if stage == "notes":
        # notes is optional; still ask once, then controller will mark done
        return "ask_notes"

    return None

# ---------------------------
# Chat (LLM-first orchestration)
# ---------------------------

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    """
    LLM-first sandwich:
      1) Planner (classify/route)
      2) Retrieval (if needed)
      3) Final composer — with controller (capture.py) owning the flow/stage via state,
         and the LLM phrasing the response using compact hints.
    """
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="empty_question")

    session_id = req.session_id
    now = _now_utc()

    # record the user turn so both heuristics & LLM see the latest message
    append_turn(session_id, "user", q)

    # ----- 1) Opportunistic short-term memory (light harvest) -----
    state = get_state(session_id)  # durable JSONB state (merged with defaults)

    # opportunistically harvest user-provided PII (never overwrite with None)
    name  = harvest_name(q)   or state.get("name")
    phone = harvest_phone(q)  or state.get("phone")
    email = harvest_email(q)  or state.get("email")
    set_state(session_id, name=name, phone=phone, email=email)

    # Read guards / tunables
    already_started = bool(state.get("lead_started_at"))
    already_done    = bool(state.get("lead_done_at"))
    nudge_at_iso    = state.get("lead_nudge_at")
    nudge_count     = int(state.get("lead_nudge_count") or 0)

    LEAD_NUDGE_COOLDOWN_SEC = getattr(CFG, "LEAD_NUDGE_COOLDOWN_SEC", 60)
    LEAD_MAX_NUDGES         = getattr(CFG, "LEAD_MAX_NUDGES", 2)

    ASK_COOLDOWN_NAME_SEC    = getattr(CFG, "ASK_COOLDOWN_NAME_SEC", 45)
    ASK_COOLDOWN_CONTACT_SEC = getattr(CFG, "ASK_COOLDOWN_CONTACT_SEC", 45)
    ASK_COOLDOWN_TIME_SEC    = getattr(CFG, "ASK_COOLDOWN_TIME_SEC", 45)
    ASK_COOLDOWN_NOTES_SEC   = getattr(CFG, "ASK_COOLDOWN_NOTES_SEC", 45)

    recent_nudge = not _cooldown_ok(nudge_at_iso, LEAD_NUDGE_COOLDOWN_SEC)

    # ----- 2) Controller (logic owner): advance/persist state, then decide hint -----
    hint: Optional[str] = None

    if lead_in_progress(session_id):
        # Advance the machine with the user's text (controller persists data/stage internally)
        _ = lead_turn(session_id, q)  # return value ignored on purpose (LLM will phrase)
        state = get_state(session_id)  # re-read after controller updates

        # derive the next ask from current state
        hint = _decide_hint_from_stage(state)

    else:
        # Not in progress — see if the user intent should *start* capture
        ql = q.lower()
        callback_re = r"\b(call\s*back|callback|schedule(?:\s+a)?\s+call|set\s*up(?:\s+a)?\s+call|book(?:\s+a)?\s+call|arrange(?:\s+a)?\s+call|arrange(?:\s+a)?\s*callback)\b"
        trigger = (
            bool(re.search(callback_re, ql))
            or any(kw in ql for kw in ["start", "begin", "i want to start", "how do i start", "i'm ready", "let's go"])
            or bool(phone or email)  # user volunteered contact — good signal to start
        )

        can_nudge = (not already_started) and (not already_done) and (not recent_nudge) and (nudge_count < LEAD_MAX_NUDGES)
        if trigger and can_nudge:
            # Start the flow (controller sets stage='name' and persists)
            _ = lead_start(session_id, kind="callback")
            # mark nudge budget + timestamp + started
            set_state(
                session_id,
                lead_started_at=now.isoformat(),
                lead_nudge_at=now.isoformat(),
                lead_nudge_count=nudge_count + 1,
            )
            state = get_state(session_id)
            hint = "ask_name"

    # ----- 3) One-ask policy with cooldowns (avoid repetition) -----
    # If we intend to ask again but have just asked this field, switch to bridge_back_to_<field>.
    last_asked = _compute_last_asked(state)

    if hint and hint.startswith("ask_"):
        field = hint.split("ask_", 1)[1]  # name|contact|time|notes
        # choose proper cooldown per field
        field_cooldown = {
            "name":    ASK_COOLDOWN_NAME_SEC,
            "contact": ASK_COOLDOWN_CONTACT_SEC,
            "time":    ASK_COOLDOWN_TIME_SEC,
            "notes":   ASK_COOLDOWN_NOTES_SEC,
        }.get(field, 45)

        if last_asked == field and recently_asked(session_id, field, field_cooldown):
            # don't repeat; gently bridge back
            hint = f"bridge_back_to_{field}"
        else:
            # stamp that we are asking this field now
            mark_asked(session_id, field)

    # If stage is done, ensure we signal a one-shot close to the UI
    end_session = False
    st_after = get_state(session_id) or {}
    if st_after.get("lead_just_done"):
        end_session = True
        # clear the one-shot flag so subsequent turns don’t force-close
        set_state(session_id, lead_just_done=False)

    # ----- 4) Facts & context assembly for the final LLM -----
    # Company facts
    fact_names = ["contact_email", "contact_phone", "contact_url", "office_address",
                  "pricing_bullet", "pricing_overview"]
    facts = get_facts(fact_names) or {}
    contact_lines = []
    if facts.get("contact_email"):  contact_lines.append(f"Email: {facts['contact_email']}")
    if facts.get("contact_phone"):  contact_lines.append(f"Phone: {facts['contact_phone']}")
    if facts.get("contact_url"):    contact_lines.append(f"Website: {facts['contact_url']}")
    if facts.get("office_address"): contact_lines.append(f"Office: {facts['office_address']}")
    contact_context = "\n".join(contact_lines) if contact_lines else "None available"

    pricing_context = ""
    if facts.get("pricing_overview"):
        pricing_context += f"Overview: {facts['pricing_overview']}\n"
    if facts.get("pricing_bullet"):
        pricing_context += f"Key point: {facts['pricing_bullet']}\n"

    # User details block for the model (never echoed as company info)
    # Use fresh state after any controller updates
    state = get_state(session_id) or {}
    user_details = (
        f"Name: {state.get('name') or '-'}\n"
        f"Phone: {state.get('phone') or '-'}\n"
        f"Email: {state.get('email') or '-'}\n"
        f"Preferred time: {state.get('preferred_time') or '-'}"
    )

    # Hints passed as plain lines in [Context] for generator.py to read
    hint_lines = ""
    if hint:
        hint_lines += f"Lead hint: {hint}\n"

    last_asked_field = _compute_last_asked(state)
    if last_asked_field:
        hint_lines += f"last_asked: {last_asked_field}\n"

    augmented_q = (
        f"{q}\n\n"
        f"[Context]\n"
        f"- User details:\n{user_details}\n"
        f"- Company contact:\n{contact_context or 'None'}\n"
        f"- Pricing:\n{pricing_context or 'None'}\n"
        f"{hint_lines}"
        f"[End Context]\n"
    )

    # ----- 5) LLM sandwich (planner -> retrieval -> final) -----
    result = generate_answer(
        question=augmented_q,
        k=getattr(CFG, "RETRIEVAL_TOP_K", 5) if (req.k is None) else req.k,
        max_context_chars=req.max_context or 3000,
        debug=req.debug,
        show_citations=req.citations,
    )

    answer = (result.get("answer") or "").strip()
    append_turn(session_id, "assistant", answer)

    return ChatResponse(
        answer=answer,
        citations=result.get("citations"),
        debug=result.get("debug"),
        end_session=end_session,
    )

# ---------------------------
# (Optional) Heuristic classifier kept for future analytics
# ---------------------------

def classify_lead_intent(turns: List[dict]) -> dict:
    text = " ".join(t.get("content", "") for t in turns[-4:])
    t = text.lower()
    explicit_cta_terms = [
        "callback", "call back",
        "book a call", "schedule a call", "set up a call", "arrange a call",
        "call me", "can you call",
        "how do i start", "let's start", "get started", "sign me up", "move forward",
        "can we talk"
    ]
    buying_terms = [
        "i like this", "sounds good", "i’m interested", "i am interested",
        "this is good", "want this", "we want this", "we need this"
    ]
    contact_given = ("@" in t) or any(k in t for k in ["phone", "call me on", "+", "whatsapp"])
    explicit_cta = any(kw in t for kw in explicit_cta_terms)
    buying = any(kw in t for kw in buying_terms)
    if explicit_cta:
        return {"interest": "explicit_cta", "explicit_cta": True, "contact_given": contact_given, "confidence": 0.9}
    if contact_given:
        return {"interest": "buying", "explicit_cta": False, "contact_given": True, "confidence": 0.8}
    if buying:
        return {"interest": "buying", "explicit_cta": False, "contact_given": False, "confidence": 0.7}
    return {"interest": "curious", "explicit_cta": False, "contact_given": False, "confidence": 0.4}