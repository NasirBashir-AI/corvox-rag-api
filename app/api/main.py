"""
app/api/main.py

FastAPI entrypoint for Corah.
- /api/health  : liveness
- /api/search  : retrieval-only probe
- /api/chat    : routed chat (smalltalk / facts-first / RAG answer)

This module wires the lightweight router and the generation pipeline.
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
from app.api.intents import detect_intent
from app.core.config import (
    ENABLE_SMALLTALK,
    ENABLE_FACTS,
    RETRIEVAL_TOP_K,
)
from app.retrieval.retriever import search, get_facts
from app.generation.generator import generate_answer
from app.lead.capture import in_progress as lead_in_progress, start as lead_start, take_turn as lead_turn
from app.lead.capture import harvest_email, harvest_phone, harvest_name
import re
from app.retrieval.leads import save_lead
from app.core.session_mem import get_state, set_state

_EMAIL_RX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RX = re.compile(r"\+?\d[\d\s().-]{6,}")


app = FastAPI(
    title="Corah API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

# CORS (relaxed by default; tighten for prod domain(s))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your site origin in prod
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
# Chat
# ---------------------------

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    """
    Routed chat:
      - smalltalk → immediate friendly reply (no RAG)
      - contact/pricing → facts-first (if enabled), fallback to generator
      - lead        → multi-turn capture flow (name → contact → time → notes)
      - other/services → generator with RAG
    """
    # 1) normalize & guard
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="empty_question")

    # 2) continue any in-progress lead flow *before* intent detection
    if lead_in_progress(req.session_id):
        return ChatResponse(answer=lead_turn(req.session_id, q))

    # 3) detect intent
    intent, smalltalk = detect_intent(q)

    # 4) smalltalk short-circuit
    if intent == "smalltalk" and ENABLE_SMALLTALK and smalltalk:
        return ChatResponse(answer=smalltalk)

    # 5) start a new lead-capture flow on explicit ask
    if intent == "lead":
        first = lead_start(req.session_id, kind="callback")
        return ChatResponse(answer=first)

    # 6) facts-first for contact/pricing
    if intent in ("contact", "pricing") and ENABLE_FACTS:
        names = (
            ["contact_email", "contact_phone", "contact_url", "office_address"]
            if intent == "contact"
            else ["pricing_bullet", "pricing_overview"]
        )
        facts = get_facts(names)
        if facts:
            if intent == "contact":
                # your existing focused contact logic...
                intent, focus = detect_intent(q)

                # --- Lead capture: harvest + short-term memory + persist ---
                state = get_state(req.session_id)

                # 1) harvest from current message (fall back to what we remembered)
                name  = harvest_name(q)   or state.get("name")
                phone = harvest_phone(q)  or state.get("phone")
                email = harvest_email(q)  or state.get("email")

                # keep short-term memory updated for this session
                set_state(req.session_id, name=name, phone=phone, email=email)

                # 2) persist a lead as soon as we have *any* contact point
                if phone or email:
                    try:
                        save_lead(
                            session_id=req.session_id,
                            name=name,
                            phone=phone,
                            email=email,
                            preferred_time=state.get("preferred_time"),
                            notes=q,
                            source="chat",
                        )
                    except Exception as e:
                        print("lead insert failed:", e)
                
                # Pure info-seeking fallback: user asked *how to contact* but didn't give their details
                if not phone and not email and focus == "generic":
                    email_f = facts.get("contact_email")
                    phone_f = facts.get("contact_phone")
                    url_f   = facts.get("contact_url")
                    addr_f  = facts.get("office_address")

                    parts = []
                    if email_f: parts.append(f"email {email_f}")
                    if phone_f: parts.append(f"phone {phone_f}")
                    if url_f:   parts.append(f"website {url_f}")
                    if addr_f:  parts.append(f"office {addr_f}")

                    if parts:
                        lead_txt, last = (", ".join(parts[:-1]), parts[-1]) if len(parts) > 1 else ("", parts[0])
                        msg = f"You can contact us via {lead_txt + (', ' if lead_txt else '')}{last}."
                        return ChatResponse(
                            answer=msg + " If you'd like, I can also arrange a callback—what's the best phone number for you?"
                        )

                # 3) determine next, natural question (no LLM here)
                if not name:
                    return ChatResponse(answer="Got it. May I take your name?")
                if not phone and not email:
                    return ChatResponse(answer="Happy to arrange a callback. What’s the best phone number for you?")
                if not phone and email:
                    return ChatResponse(answer="Thanks for the email. Do you also have a phone number for the callback?")

                # 4) we have enough to proceed; confirm succinctly
                if phone and email:
                    return ChatResponse(answer=f"Thanks {name}. I’ve noted your phone and email. We’ll be in touch shortly.")
                if phone:
                    return ChatResponse(answer=f"Thanks {name}. I’ve got your phone number. We’ll call you shortly.")
                # fallback: only email (no phone)
                return ChatResponse(answer=f"Thanks {name}. I’ve noted your email. If you’d like a call, feel free to share a phone number.")
            
            else:
                # pricing branch
                bullets_val = facts.get("pricing_bullet")
                overview    = facts.get("pricing_overview")
                top = [bullets_val] if bullets_val else []
                joined = " • ".join(top) if top else ""
                if overview or joined:
                    prefix = f"{overview} — " if overview else ""
                    return ChatResponse(answer=f"{prefix}{joined}")

    # 7) default: generate with RAG (never deflect)
    result = generate_answer(
        question=q,
        k=req.k or RETRIEVAL_TOP_K,
        max_context_chars=req.max_context or 3000,
        debug=req.debug,
        show_citations=req.citations,
    )
    return ChatResponse(**result)  # type: ignore[arg-type]