"""
app/api/main.py

FastAPI entrypoint for Corah.
- /api/health  : liveness
- /api/search  : retrieval-only probe
- /api/chat    : LLM-first sandwich (context -> retrieval -> LLM)
"""

from __future__ import annotations
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SearchHit,
    SearchResponse,
)
from app.core.config import RETRIEVAL_TOP_K
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
from app.core.session_mem import get_state, set_state, append_turn, recent_turns


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

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    """
    LLM-first sandwich on every turn:
      user -> LLM (sees state + facts + last msg) -> retrieval -> LLM final
    """
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="empty_question")

    session_id = req.session_id

    # record the user turn so heuristics/LLM see the latest message
    append_turn(session_id, "user", q)

    # ----- 1) Short-term memory (harvest & store) -----
    state = get_state(session_id)  # { name, phone, email, preferred_time, ... }
    name  = harvest_name(q)   or state.get("name")
    phone = harvest_phone(q)  or state.get("phone")
    email = harvest_email(q)  or state.get("email")
    set_state(session_id, name=name, phone=phone, email=email)

    # ----- 2) Lead flow hinting for the LLM -----
    lead_hint = None
    if lead_in_progress(session_id):
        next_prompt = lead_turn(session_id, q)  # advances stage; returns the next question to ask
        lead_hint = f"Lead flow in progress. Ask this next, in a warm, concise way: {next_prompt}"
    else:
        trigger = any(x in q.lower() for x in [
            "start", "begin", "book a call", "call me", "can you call", "arrange a call",
            "i want to start", "how do i start", "i'm ready", "let's go"
        ]) or bool(phone or email)
        if trigger:
            first = lead_start(session_id, kind="callback")  # sets stage="name"
            lead_hint = f"Start a lead capture (callback). Ask this first: {first}"

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
    lead_state_line = f"name={name or '-'}, phone={phone or '-'}, email={email or '-'}, preferred_time={state.get('preferred_time','-')}"
    lead_hint_text = f"\nLead hint: {lead_hint}" if lead_hint else ""

    augmented_q = (
        f"{q}\n\n"
        f"[Context]\n"
        f"- Company contact:\n{contact_context or 'None'}\n"
        f"- Pricing:\n{pricing_context or 'None'}\n"
        f"- Lead state: {lead_state_line}\n"
        f"{lead_hint_text}\n"
        f"[End Context]\n"
    )

    # ----- 5) LLM sandwich (retrieve -> LLM) -----
    result = generate_answer(
        question=augmented_q,
        k=req.k or RETRIEVAL_TOP_K,
        max_context_chars=req.max_context or 3000,
        debug=req.debug,
        show_citations=req.citations,
    )

    # ----- 6) Behavior-triggered lead prompt (post-LLM) -----
    answer = result.get("answer", "").strip()
    cls = classify_lead_intent(recent_turns(session_id, n=4))

    trigger2 = (
        cls.get("explicit_cta")
        or cls.get("contact_given")
        or (cls.get("interest") in ("buying", "explicit_cta") and float(cls.get("confidence", 0)) >= 0.65)
    )
    if trigger2 and not lead_in_progress(session_id):
        lead_msg = lead_start(session_id)  # “Great — I can arrange that. What’s your name?”
        answer = (answer + "\n\n" + lead_msg).strip()

    append_turn(session_id, "assistant", answer)
    return ChatResponse(answer=answer, citations=result.get("citations"), debug=result.get("debug"))


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
        "book a call", "arrange a call", "call me", "can you call", "call back",
        "how do i start", "let's start", "get started", "sign me up", "move forward",
        "can we talk", "schedule a call", "set up a call"
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