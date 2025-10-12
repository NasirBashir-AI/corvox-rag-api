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
)
from app.core.session_mem import (
    get_state, set_state, append_turn, recent_turns,
    get_pending_offer, set_pending_offer, clear_pending_offer,
    set_current_topic, update_summary,
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
# Tiny intent + topic helpers (surgical, non-invasive)
# ---------------------------

_AFFIRM = re.compile(r"\b(yes|yep|yeah|sure|ok|okay|go ahead|please do|tell me more|sounds good|great|let's do it)\b", re.I)
_DECLINE = re.compile(r"\b(no|nah|nope|not now|later|don'?t|do not|skip)\b", re.I)

def _is_affirm(text: str) -> bool:
    return bool(_AFFIRM.search(text or ""))

def _is_decline(text: str) -> bool:
    return bool(_DECLINE.search(text or ""))

def _looks_like_followup_request(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in [
        "tell me more", "explain", "how does", "how it", "how this", "details", "more about", "what next", "what's next",
        "how it integrates", "how integrate", "integration", "walk me through", "how would it work"
    ])

def _extract_topic(text: str) -> Optional[str]:
    t = (text or "").lower()
    if "whatsapp" in t and "chatbot" in t:
        return "WhatsApp chatbot"
    if "reconciliation" in t:
        return "AI reconciliation"
    if "lead" in t and "capture" in t:
        return "Lead capture"
    if "student" in t and "recruit" in t:
        return "Student recruitment AI"
    if "jewell" in t or "jewel" in t:
        return "Jewellery retail AI"
    if "account" in t:
        return "Accounting automation"
    return None

# Detect if the assistant made an offer/promise that should be fulfilled on user affirmation.
_OFFER_RE = re.compile(
    r"(would you like (to know more|me to (explain|walk you through)|details)|"
    r"want me to (explain|go over)|shall i (explain|detail)|"
    r"should i (send|share) (details|information))",
    re.I
)

def _detect_offer_tag(answer: str) -> Optional[str]:
    if not answer:
        return None
    if _OFFERS := _OFFER_RE.search(answer):
        # keep a short generic tag; we don’t need to be fancy here
        return "promised_explanation"
    return None

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

    # ----- 1) Opportunistic short-term memory (light harvest) -----
    state = get_state(session_id)
    name  = harvest_name(q)   or state.get("name")
    phone = harvest_phone(q)  or state.get("phone")
    email = harvest_email(q)  or state.get("email")
    set_state(session_id, name=name, phone=phone, email=email)

    # Topic (very light, only when obvious)
    topic = _extract_topic(q)
    if topic:
        set_current_topic(session_id, topic)

    # Track guards
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

    # Update rolling summary with a simple intent label
    intent_label = "affirm" if _is_affirm(q) else ("decline" if _is_decline(q) else ("followup" if _looks_like_followup_request(q) else "ask"))
    summary = update_summary(session_id, intent=intent_label)

    # ----- 2) Controller signals (with pending-offer precedence) -----
    hint_str: Optional[str] = None
    pending = get_pending_offer(session_id)

    if pending and (_is_affirm(q) or _looks_like_followup_request(q)):
        # Honor the earlier offer first; pause capture asks for this turn.
        hint_str = "resolve_pending_offer"
    else:
        # Normal lead flow
        if lead_in_progress(session_id):
            sig = lead_turn(session_id, q)  # returns dict with {"hint": "..."} in your current controller
            if isinstance(sig, dict) and sig.get("hint"):
                hint_str = sig["hint"]
        else:
            # CTA trigger
            ql = q.lower()
            callback_re = r"\b(call\s*back|callback|schedule(?:\s+a)?\s+call|set\s*up(?:\s+a)?\s+call|book(?:\s+a)?\s+call|arrange(?:\s+a)?\s+call|arrange(?:\s+a)?\s*callback)\b"
            trigger = (
                bool(re.search(callback_re, ql))
                or any(kw in ql for kw in ["start", "begin", "i want to start", "how do i start", "i'm ready", "let's go"])
                or bool(phone or email)
            )
            can_nudge = (not already_started) and (not already_done) and (not recent_nudge) and (nudge_count < LEAD_MAX_NUDGES)
            if trigger and can_nudge:
                sig = lead_start(session_id, kind="callback")  # controller sets stage and returns {"hint":"ask_name"}
                if isinstance(sig, dict) and sig.get("hint"):
                    hint_str = sig["hint"]
                set_state(
                    session_id,
                    lead_started_at=now.isoformat(),
                    lead_nudge_at=now.isoformat(),
                    lead_nudge_count=nudge_count + 1,
                )

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
    # user details
    st_now = get_state(session_id)  # re-read to include any updates above
    user_details = (
        f"Name: {st_now.get('name') or '-'}\n"
        f"Phone: {st_now.get('phone') or '-'}\n"
        f"Email: {st_now.get('email') or '-'}\n"
        f"Preferred time: {st_now.get('preferred_time') or '-'}"
    )

    # asked recency (optional; if you keep it)
    last_asked = None
    ts_map = {}
    for field in ("name", "contact", "time", "notes"):
        t_iso = st_now.get(f"asked_for_{field}_at")
        if t_iso:
            try:
                ts_map[field] = datetime.fromisoformat(t_iso)
            except Exception:
                pass
    if ts_map:
        last_asked = max(ts_map.items(), key=lambda kv: kv[1])[0]

    lead_hint_line = f"Lead hint: {hint_str}" if hint_str else ""
    last_asked_line = f"last_asked: {last_asked}" if last_asked else ""
    summary_block = st_now.get("session_summary") or ""
    current_topic = st_now.get("current_topic") or ""

    hint_lines = ""
    if lead_hint_line:
        hint_lines += lead_hint_line + "\n"
    if last_asked_line:
        hint_lines += last_asked_line + "\n"

    augmented_q = (
        f"{q}\n\n"
        f"[Context]\n"
        f"- Summary:\n{summary_block or 'None'}\n"
        f"- Current topic: {current_topic or '-'}\n"
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

    # Manage pending offer tag based on the assistant's reply
    if hint_str == "resolve_pending_offer":
        clear_pending_offer(session_id)
    else:
        offer_tag = _detect_offer_tag(answer)
        if offer_tag:
            set_pending_offer(session_id, offer_tag)

    # One-shot session close if we JUST finished the lead
    end_session = False
    st_after = get_state(session_id) or {}
    if st_after.get("lead_just_done"):
        end_session = True
        set_state(session_id, lead_just_done=False)

    return ChatResponse(
        answer=answer,
        citations=result.get("citations"),
        debug=result.get("debug"),
        end_session=end_session,
    )