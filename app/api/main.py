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
from typing import List, Tuple
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import (
    ChatRequest, ChatResponse, HealthResponse, SearchHit, SearchResponse,
)
from app.core.config import RETRIEVAL_TOP_K
from app.core.utils import normalize_ws
from app.retrieval.retriever import search
from app.generation.generator import generate_answer
from app.core.session_mem import (
    get_state, set_state, append_turn, recent_turns, update_summary, get_lead_slots,
)
from app.api.intents import detect_intent, smalltalk_reply
from app.lead.capture import update_lead_info, next_lead_question

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
# Chat orchestration (lean controller + lead capture)

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    q_raw = (req.question or "").strip()
    if not q_raw:
        raise HTTPException(status_code=400, detail="empty_question")
    session_id = (req.session_id or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="missing_session_id")

    # Normalize whitespace to stabilize intent routing
    q = normalize_ws(q_raw)

    # Ensure session exists
    st = get_state(session_id) or {}
    if not st:
        st = {"created_at": _now_iso(), "summary": "", "turns": []}
        set_state(session_id, **st)

    # Record the user turn
    append_turn(session_id, role="user", content=q)

    # Opportunistic lead extraction (never overwrites a confirmed value)
    slots = update_lead_info(session_id, q)

    # Intent routing (info-first)
    kind, topic = detect_intent(q)

    # Smalltalk is answered immediately (no RAG)
    if kind == "smalltalk":
        answer = smalltalk_reply(q)
        append_turn(session_id, role="assistant", content=answer)
        try:
            update_summary(session_id)
        except Exception:
            pass
        return ChatResponse(answer=answer, citations=None, debug=None)

    # Build structured context (include lead slots as [User details])
    st2 = get_state(session_id) or st
    summary_txt = st2.get("summary", "")
    last_turns = recent_turns(session_id, n=6)

    def _fmt_slots(s: dict) -> str:
        # Only show fields we have; keep it compact.
        order = ("name", "company", "email", "phone", "time")
        labels = {
            "name": "name", "company": "company",
            "email": "email", "phone": "phone", "time": "preferred_time",
        }
        lines = []
        for k in order:
            v = s.get(k)
            if v:
                lines.append(f"- {labels[k]}: {v}")
        return "\n".join(lines) if lines else "None"

    user_details_block = _fmt_slots(get_lead_slots(session_id))

    context_block = (
        "[Context]\n"
        f"- Summary: {summary_txt or 'None'}\n"
        f"- Recent turns:\n" + "\n".join(
            f"  - {t.get('role','?')}: {t.get('content','')}" for t in last_turns
        ) + "\n"
        f"- User details:\n{user_details_block}\n"
        "[Intent]\n"
        f"kind: {kind}\n"
        f"topic: {topic or 'None'}\n"
        "[End Context]\n"
    )

    # Ask the generator (help-first, KB-first; will honor [Intent] + [User details])
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

    # Gated, single follow-up capture question (optional)
    st3 = get_state(session_id) or {}
    turn_idx = int(st3.get("turns_count", 0))  # maintained by append_turn
    ask = next_lead_question(
        session_id=session_id,
        turn_idx=turn_idx,
        user_intent=(topic or kind or ""),
    )
    if ask:
        # Append politely on a new paragraph
        if answer_text:
            answer_text = f"{answer_text}\n\n{ask}"
        else:
            answer_text = ask

    # Record assistant turn & update summary
    append_turn(session_id, role="assistant", content=answer_text)
    try:
        update_summary(session_id)
    except Exception:
        pass

    return ChatResponse(answer=answer_text, citations=citations, debug=dbg)