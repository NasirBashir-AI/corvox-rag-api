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
      - other/services → generator with RAG
    """
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="empty_question")

    intent, smalltalk = detect_intent(q)

    # 1) Smalltalk short-circuit
    if intent == "smalltalk" and ENABLE_SMALLTALK and smalltalk:
        return ChatResponse(answer=smalltalk)

    # 2) Facts-first for contact/pricing
    if intent in ("contact", "pricing") and ENABLE_FACTS:
        names = ["contact_email", "contact_phone", "contact_url", "office_address"] if intent == "contact" else ["pricing_bullet", "pricing_overview"]
        facts = get_facts(names)
        if facts:
           if intent == "contact":
                # second value from detect_intent now carries the focus (email/phone/address/url/generic)
                _intent, focus = detect_intent(q)

                # Decide which facts to fetch based on the focus
                if focus == "email":
                    names = ["contact_email"]
                elif focus == "phone":
                    names = ["contact_phone"]
                elif focus == "address":
                    names = ["office_address"]
                elif focus == "url":
                    names = ["contact_url"]
                else:  # generic contact
                    names = ["contact_email", "contact_phone", "contact_url", "office_address"]

                facts = get_facts(names)
                if facts:
                    email = facts.get("contact_email")
                    phone = facts.get("contact_phone")
                    url   = facts.get("contact_url")
                    addr  = facts.get("office_address")

                    # Focused, natural sentences
                    if focus == "email" and email:
                        return ChatResponse(answer=f"You can email us at {email}.")
                    if focus == "phone" and phone:
                        return ChatResponse(answer=f"You can call us on {phone}.")
                    if focus == "address" and addr:
                        return ChatResponse(answer=f"We’re based at {addr}.")
                    if focus == "url" and url:
                        return ChatResponse(answer=f"Our website is {url}.")

                    # Generic contact: include whatever exists, phrased nicely
                    parts = []
                    if email: parts.append(f"email ({email})")
                    if phone: parts.append(f"phone")
                    if url:   parts.append(f"our website: {url}")
                    if addr:  parts.append(f"our office: {addr}")

                    if parts:
                        # e.g., "You can contact us via email (x), phone, or our website: y. Our office: …"
                        if len(parts) == 1:
                            return ChatResponse(answer=f"You can contact us via {parts[0]}.")
                        else:
                            lead, last = ", ".join(parts[:-1]), parts[-1]
                            return ChatResponse(answer=f"You can contact us via {lead}, or {last}.")

                    # If we have no facts at all, fall back politely
                    return ChatResponse(answer="You can reach us through our website or messaging channels.")

            else:
                # pricing
                bullets_val = facts.get("pricing_bullet")       # string or None
                overview    = facts.get("pricing_overview")     # string or None

                # Show at most one “bullet” string if present (safe even if it’s a single string)
                top = [bullets_val] if bullets_val else []
                joined = " • ".join(top) if top else ""

                if overview or joined:
                    prefix = f"{overview} — " if overview else ""
                    return ChatResponse(answer=f"{prefix}{joined}")
                # otherwise fall through to generator

    # 3) Default: generate with RAG (never deflect)
    result = generate_answer(
        question=q,
        k=req.k or RETRIEVAL_TOP_K,
        max_context_chars=req.max_context or 3000,
        debug=req.debug,
        show_citations=req.citations,
    )
    return ChatResponse(**result)  # type: ignore[arg-type]