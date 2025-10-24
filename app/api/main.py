# app/api/main.py
"""
FastAPI entrypoint for Corah.

Endpoints
- GET  /api/health
- GET  /api/search
- GET  /api/ping
- POST /api/chat
"""

from __future__ import annotations
from typing import List
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import (
    ChatRequest, ChatResponse, HealthResponse, SearchHit, SearchResponse,
)
from app.core.config import RETRIEVAL_TOP_K
from app.core.utils import normalize_ws
from app.retrieval.retriever import search, get_facts
from app.generation.generator import generate_answer
from app.core.session_mem import (
    get_state, set_state, append_turn, recent_turns, update_summary,
    get_lead_slots,
)
from app.api.intents import detect_intent, smalltalk_reply
from app.lead.capture import update_lead_info

# -------------------------------------------------------------------

app = FastAPI(
    title="Corah API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# -------------------------------------------------------------------
# Health

@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True)

# -------------------------------------------------------------------
# Retrieval probe

@app.get("/api/search", response_model=SearchResponse)
def api_search(
    q: str = Query(..., min_length=1),
    k: int = Query(RETRIEVAL_TOP_K, ge=1, le=20),
) -> SearchResponse:
    try:
        raw = search(q, k=k)
        hits: List[SearchHit] = [
            SearchHit(
                document_id=h.get("document_id"),
                chunk_no=h.get("chunk_no"),
                title=h.get("title"),
                source_uri=h.get("source_uri"),
                content=h.get("content"),
                score=float(h.get("score", 0.0)),
            )
            for h in raw
        ]
        return SearchResponse(hits=hits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search_failed: {type(e).__name__}: {e}")

# -------------------------------------------------------------------
# Ping (heartbeat used by the web UI)

@app.get("/api/ping")
def api_ping(session_id: str = Query(..., min_length=1)) -> dict:
    st = get_state(session_id) or {}
    set_state(session_id, **{**st, "last_ping": _now_iso()})
    return {"ok": True}

# -------------------------------------------------------------------
# Chat orchestration (lean controller)

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    q_raw = (req.question or "").strip()
    if not q_raw:
        raise HTTPException(status_code=400, detail="empty_question")
    session_id = (req.session_id or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="missing_session_id")

    q = normalize_ws(q_raw)

    # Ensure session shell
    st = get_state(session_id) or {}
    if not st:
        st = {"created_at": _now_iso(), "summary": "", "turns": []}
        set_state(session_id, **st)

    # Record user turn
    append_turn(session_id, role="user", content=q)

    # Opportunistic lead extraction (name/email/phone/time/company)
    update_lead_info(session_id, q)

    # Intent routing
    kind, topic = detect_intent(q)

    # Smalltalk: immediate reply, then update summary
    if kind == "smalltalk":
        answer = smalltalk_reply(q)
        append_turn(session_id, role="assistant", content=answer)
        try:
            update_summary(session_id)
        except Exception:
            pass
        return ChatResponse(answer=answer, citations=None, debug=None)

    # Build [User details] from lead slots
    slots = get_lead_slots(session_id)
    user_details_txt = "\n".join(
        f"{k}: {v}" for k, v in (
            ("name", slots.get("name")),
            ("company", slots.get("company")),
            ("email", slots.get("email")),
            ("phone", slots.get("phone")),
            ("preferred_time", slots.get("preferred_time") or slots.get("time")),
        ) if v
    )

    # Pull company contact/pricing facts (if any)
    facts = get_facts(["contact_email", "office_address", "contact_url", "pricing_overview"])
    contact_ctx = "\n".join(
        f"{k}: {v}" for k, v in (
            ("email", facts.get("contact_email")),
            ("office_address", facts.get("office_address")),
            ("contact_url", facts.get("contact_url")),
        ) if v
    )
    pricing_ctx = facts.get("pricing_overview", "") or "Pricing depends on scope; we start with a short discovery call, then share a clear quote."

    # Recent info
    st2 = get_state(session_id) or st
    summary_txt = st2.get("summary", "")
    last_turns = recent_turns(session_id, n=6)

    # Compose the context block (explicit Intent)
    context_block = (
        "[Context]\n"
        f"- Summary: {summary_txt or 'None'}\n"
        f"- Recent turns:\n" + "\n".join(
            f"  - {t.get('role','?')}: {t.get('content','')}" for t in last_turns
        ) + "\n"
        f"- User details:\n{user_details_txt or 'None'}\n"
        f"- Company contact:\n{contact_ctx or 'None'}\n"
        f"- Pricing:\n{pricing_ctx or 'None'}\n"
        "[Intent]\n"
        f"kind: {kind}\n"
        f"topic: {topic or 'None'}\n"
        "[End Context]\n"
    )

    # Ask the generator (KB-first, answer-first)
    try:
        gen = generate_answer(
            question=f"{q}\n\n{context_block}",
            k=RETRIEVAL_TOP_K,
            max_context_chars=3000,
            debug=False,
            show_citations=True,
        )
    except Exception as e:
        append_turn(session_id, role="assistant", content=f"(internal error: {e})")
        raise HTTPException(status_code=500, detail=f"generation_failed: {type(e).__name__}: {e}")

    answer_text = (gen.get("answer") or "").strip()
    citations = gen.get("citations") or None
    dbg = gen.get("debug") or None

    append_turn(session_id, role="assistant", content=answer_text)
    try:
        update_summary(session_id)
    except Exception:
        pass

    return ChatResponse(answer=answer_text, citations=citations, debug=dbg)