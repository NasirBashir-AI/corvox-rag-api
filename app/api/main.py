"""
app/api/main.py

FastAPI entrypoint for Corah.
- /api/health  : liveness
- /api/search  : retrieval-only probe
- /api/chat    : LLM-first sandwich (planner -> retrieval -> final)
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
from app.core.config import RETRIEVAL_TOP_K, LEAD_MAX_NUDGES
from app.retrieval.retriever import search, get_facts
from app.generation.generator import generate_answer
from app.lead.capture import (
    in_progress as lead_in_progress,
    start as lead_start,
    next_hint as lead_next_hint,   # <- signals (ask=name, bridge_back_to_time, etc.)
    harvest_email,
    harvest_phone,
    harvest_name,
)
from app.core.session_mem import (
    get_state, set_state, append_turn, recent_turns,
    bump_nudge,  # increments lead_nudge_count and sets lead_nudge_at
)

app = FastAPI(
    title="Corah API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

# CORS (relax for dev; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
def api_search(q: str = Query(..., min_length=1), k: int = Query(RETRIEVAL_TOP_K, ge=1, le=20)) -> SearchResponse:
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
# Chat (LLM-first sandwich)
# ---------------------------

_COOLDOWN_SECONDS = 60  # nudger cooldown

def _compute_last_asked_field(state: dict) -> Optional[str]:
    """
    From asked_for_*_at timestamps, return the most recently asked field.
    """
    fields = ["name", "phone", "email", "time", "notes"]
    latest = None
    latest_ts = None
    for f in fields:
        ts = state.get(f"asked_for_{f}_at")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            continue
        if latest_ts is None or dt > latest_ts:
            latest_ts = dt
            latest = f
    return latest

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    """
    Clean pipeline each turn:
      user -> (update memory) -> (maybe start lead via nudger) -> controller hint -> LLM planner -> retrieval -> LLM final
    No direct text appends from code; we pass only intent signals to the model.
    """
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="empty_question")

    session_id = req.session_id
    now = datetime.now(timezone.utc)

    # 0) record the user turn so heuristics/LLM see the latest message
    append_turn(session_id, "user", q)

    # 1) harvest light PII into session memory (fast heuristics)
    state = get_state(session_id)  # includes lead_* flags, asked_for_*_at, turns, etc.
    name  = harvest_name(q)   or state.get("name")
    phone = harvest_phone(q)  or state.get("phone")
    email = harvest_email(q)  or state.get("email")
    set_state(session_id, name=name, phone=phone, email=email)

    # guard flags & nudge window
    state = get_state(session_id)  # refresh
    started_at_iso   = state.get("lead_started_at")
    done_at_iso      = state.get("lead_done_at")
    nudge_at_iso     = state.get("lead_nudge_at")
    nudge_count      = int(state.get("lead_nudge_count") or 0)

    already_started  = bool(started_at_iso)
    already_done     = bool(done_at_iso)

    recent_nudge = False
    if nudge_at_iso:
        try:
            last_dt = datetime.fromisoformat(nudge_at_iso)
            recent_nudge = (now - last_dt) < timedelta(seconds=_COOLDOWN_SECONDS)
        except Exception:
            recent_nudge = False

    # 2) Decide whether to gently start capture (nudger) — controller will own stages
    # Use classifier + simple phone/email presence
    cls = classify_lead_intent(recent_turns(session_id, n=4))
    intent_trigger = (
        cls.get("explicit_cta")
        or cls.get("contact_given")
        or (cls.get("interest") in ("buying", "explicit_cta") and float(cls.get("confidence", 0)) >= 0.65)
        or bool(phone or email)
    )

    if (not lead_in_progress(session_id)) and (not already_started) and (not already_done):
        if intent_trigger and (not recent_nudge) and (nudge_count < LEAD_MAX_NUDGES):
            # Start the flow (logic only). Ignore returned copy; LLM will phrase the ask.
            lead_start(session_id, kind="callback")
            bump_nudge(session_id, now.isoformat())  # sets lead_nudge_at and increments count
            # also stamp lead_started_at to prevent re-starts
            set_state(session_id, lead_started_at=now.isoformat())

    # 3) Get controller signal for this turn (no copy, only 'hint')
    lead_hint = None
    if lead_in_progress(session_id):
        sig = lead_next_hint(session_id, q) or {}
        hint = sig.get("hint")
        if hint:
            lead_hint = f"ask={hint}"  # e.g., ask=ask_name, ask=ask_phone_or_email, ask=bridge_back_to_time

    # 4) Gather company facts for the LLM
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

    # 5) Compose augmented question (pass only signals, not copy)
    state = get_state(session_id)  # latest snapshot
    user_details = (
        f"Name: {state.get('name','-')}\n"
        f"Phone: {state.get('phone','-')}\n"
        f"Email: {state.get('email','-')}\n"
        f"Preferred time: {state.get('preferred_time','-')}"
    )
    last_asked = _compute_last_asked_field(state) or "-"
    lead_hint_text = f"\nLead hint: {lead_hint}" if lead_hint else ""
    lead_meta = f"Lead last asked: {last_asked}"

    augmented_q = (
        f"{q}\n\n"
        f"[Context]\n"
        f"- User details:\n{user_details}\n"
        f"- Company contact:\n{contact_context or 'None'}\n"
        f"- Pricing:\n{pricing_context or 'None'}\n"
        f"- {lead_meta}\n"
        f"{lead_hint_text}\n"
        f"[End Context]\n"
    )

    # 6) LLM sandwich (planner -> retrieval -> final)
    result = generate_answer(
        question=augmented_q,
        k=req.k or RETRIEVAL_TOP_K,
        max_context_chars=req.max_context or 3000,
        debug=req.debug,
        show_citations=req.citations,
    )

    answer = (result.get("answer") or "").strip()

    # 7) Save assistant turn and decide on session closure
    append_turn(session_id, "assistant", answer)

    end_session = False
    st_after = get_state(session_id) or {}
    if st_after.get("lead_just_done"):
        end_session = True
        # clear one-shot so subsequent Q&A can continue if user keeps typing
        set_state(session_id, lead_just_done=False)

    return ChatResponse(
        answer=answer,
        citations=result.get("citations"),
        debug=result.get("debug"),
        end_session=end_session,
    )

# ---------------------------
# Tiny heuristic classifier (no LLM yet)
# ---------------------------

def classify_lead_intent(turns: List[dict]) -> dict:
    """
    Looks at the last few turns and flags 'explicit_cta', 'contact_given', etc.
    """
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