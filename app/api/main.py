"""
app/api/main.py

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
from app.core.config import (
    RETRIEVAL_TOP_K,
    LEAD_NUDGE_COOLDOWN_SEC,
    LEAD_MAX_NUDGES,
)
from app.retrieval.retriever import search, get_facts
from app.generation.generator import generate_answer
from app.lead.capture import (
    in_progress as lead_in_progress,
    start as lead_start,
    take_turn as lead_turn,
    harvest_email,
    harvest_phone,
    harvest_name,
    harvest_time,   # <-- NEW
)
from app.core.session_mem import get_state, set_state, append_turn, recent_turns, update_summary  # ensure update_summary exists

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
# Chat (LLM-first orchestration)
# ---------------------------

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    """
    LLM-first sandwich:
      1) Planner LLM (classify/route)
      2) Retrieval (if needed)
      3) Final LLM (compose) — with controller (capture.py) owning the flow/stage via signals
    """
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="empty_question")

    session_id = req.session_id
    now = datetime.now(timezone.utc)

    # record the user turn so heuristics/LLM see the latest message
    append_turn(session_id, "user", q)

    # keep rolling summary fresh (industry, goal, channel, etc.)
    try:
        update_summary(session_id)  # no-op if already present; safe to call every turn
    except Exception:
        pass

    # ----- 1) Opportunistic short-term memory (light harvest) -----
    state = get_state(session_id)  # durable JSONB row
    name  = harvest_name(q)   or state.get("name")
    phone = harvest_phone(q)  or state.get("phone")
    email = harvest_email(q)  or state.get("email")
    time_str = harvest_time(q) or state.get("preferred_time")  # <-- NEW: opportunistic time capture

    set_state(session_id, name=name, phone=phone, email=email, preferred_time=time_str)

    # Guards
    already_started = bool(state.get("lead_started_at"))
    already_done    = bool(state.get("lead_done_at"))
    nudge_at_iso    = state.get("lead_nudge_at")
    nudge_count     = int(state.get("lead_nudge_count") or 0)
    recent_nudge    = False
    if nudge_at_iso:
        try:
            last_dt = datetime.fromisoformat(nudge_at_iso)
            recent_nudge = (now - last_dt) < timedelta(seconds=LEAD_NUDGE_COOLDOWN_SEC)
        except Exception:
            recent_nudge = False

    # ----- 2) Controller signals (no copy) -----
    lead_signal = None  # next prompt text (we keep your existing pattern)
    if lead_in_progress(session_id):
        lead_signal = lead_turn(session_id, q)
    else:
        # Intent trigger (richer CTA matcher)
        ql = q.lower()
        callback_re = r"\b(call\s*back|callback|schedule(?:\s+a)?\s+call|set\s*up(?:\s+a)?\s+call|book(?:\s+a)?\s+call|arrange(?:\s+a)?\s+call|arrange(?:\s+a)?\s*callback)\b"
        trigger = (
            bool(re.search(callback_re, ql))
            or any(kw in ql for kw in ["start", "begin", "i want to start", "how do i start", "i'm ready", "let's go"])
            or bool(phone or email)
            or bool(harvest_time(q))  # <-- NEW: offering a time is also intent
        )

        can_nudge = (not already_started) and (not already_done) and (not recent_nudge) and (nudge_count < LEAD_MAX_NUDGES)
        if trigger and can_nudge:
            signal = lead_start(session_id, kind="callback")  # returns first prompt text
            lead_signal = signal
            # bump nudge budget & timestamp, and mark started
            set_state(
                session_id,
                lead_started_at=now.isoformat(),
                lead_nudge_at=now.isoformat(),
                lead_nudge_count=nudge_count + 1,
            )

    # Map controller signal into a readable hint line for the generator
    hint_str: Optional[str] = None
    if isinstance(lead_signal, dict) and lead_signal.get("hint"):
        hint_str = lead_signal["hint"]
    elif isinstance(lead_signal, str) and lead_signal.strip():
        # keep legacy text prompt as a hint line for the LLM to phrase naturally
        hint_str = f"next_prompt: {lead_signal.strip()}"

    # ----- 3) Facts for the LLM (contact/pricing) -----
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

    # ----- 4) Compose augmented question for the generator -----
    # IMPORTANT: re-fetch state so preferred_time updates from harvester are included
    state = get_state(session_id)
    user_details = (
        f"Name: {state.get('name') or '-'}\n"
        f"Phone: {state.get('phone') or '-'}\n"
        f"Email: {state.get('email') or '-'}\n"
        f"Preferred time: {state.get('preferred_time') or '-'}"
    )

    # derive last_asked from the timestamps we keep in session (if any)
    last_asked: Optional[str] = None
    ts_map = {}
    for field in ("name", "contact", "time", "notes"):
        t_iso = state.get(f"asked_for_{field}_at")
        if t_iso:
            try:
                ts_map[field] = datetime.fromisoformat(t_iso)
            except Exception:
                pass
    if ts_map:
        last_asked = max(ts_map.items(), key=lambda kv: kv[1])[0]

    # Session summary + topic + recent turns (assumed maintained elsewhere)
    session_summary = state.get("session_summary") or ""
    current_topic   = state.get("current_topic") or ""
    recent = recent_turns(session_id, n=6)
    recent_lines = []
    for t in recent:
        role = t.get("role")
        content = (t.get("content") or "").strip()
        if not content:
            continue
        # keep short snippets to avoid blowing context
        if len(content) > 220:
            content = content[:220] + "..."
        recent_lines.append(f"{role}: {content}")
    recent_block = "\n".join(recent_lines)

    hint_lines = ""
    if hint_str:
        hint_lines += f"Lead hint: {hint_str}\n"
    if last_asked:
        hint_lines += f"last_asked: {last_asked}\n"

    augmented_q = (
        f"{q}\n\n"
        f"[Context]\n"
        f"- Summary:\n{session_summary or '-'}\n"
        f"- Current topic:\n{current_topic or '-'}\n"
        f"- Recent turns:\n{recent_block or '-'}\n"
        f"- User details:\n{user_details}\n"
        f"- Company contact:\n{contact_context or 'None'}\n"
        f"- Pricing:\n{pricing_context or 'None'}\n"
        f"{hint_lines}"
        f"[End Context]\n"
    )

    # ----- 5) LLM sandwich (planner -> retrieval -> final) -----
    result = generate_answer(
        question=augmented_q,
        k=req.k or RETRIEVAL_TOP_K,
        max_context_chars=req.max_context or 3000,
        debug=req.debug,
        show_citations=req.citations,
    )

    answer = (result.get("answer") or "").strip()
    append_turn(session_id, "assistant", answer)

    # One-shot session close if we JUST finished the lead
    end_session = False
    st_after = get_state(session_id) or {}
    if st_after.get("lead_just_done"):
        end_session = True
        # clear the one-shot flag so subsequent turns don’t force-close
        set_state(session_id, lead_just_done=False)

    return ChatResponse(
        answer=answer,
        citations=result.get("citations"),
        debug=result.get("debug"),
        end_session=end_session,
    )

# ---------------------------
# Tiny heuristic classifier (kept for future analytics if needed)
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