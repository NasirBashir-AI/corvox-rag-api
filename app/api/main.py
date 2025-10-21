# app/api/main.py
"""
FastAPI entrypoint for Corah.

Endpoints
- GET  /api/health  : liveness probe
- GET  /api/search  : retrieval probe
- GET  /api/ping    : lightweight heartbeat (keeps sessions warm)
- POST /api/chat    : chat orchestration (records turns, calls generator)
"""

from __future__ import annotations

from typing import List
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Schemas
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SearchHit,
    SearchResponse,
)

# Config + building blocks
from app.core.config import RETRIEVAL_TOP_K
from app.retrieval.retriever import search
from app.generation.generator import generate_answer
from app.core.session_mem import (
    get_state,
    set_state,
    append_turn,
    recent_turns,
    update_summary,
)

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Corah API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

# CORS — relaxed for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True)

# -----------------------------------------------------------------------------
# Retrieval probe
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Lightweight heartbeat (needed by the web UI)
# -----------------------------------------------------------------------------

@app.get("/api/ping")
def api_ping(session_id: str = Query(..., min_length=1)) -> dict:
    """
    Keeps the session 'warm' so the front-end doesn't show a network error.
    """
    st = get_state(session_id) or {}
    set_state(session_id, **{**st, "last_ping": _now_iso()})
    return {"ok": True}

# -----------------------------------------------------------------------------
# Chat
# -----------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    """
    Minimal, robust orchestration:
      - ensure session exists
      - record user turn
      - call generator
      - record assistant turn + update summary
      - return ChatResponse(answer=..., citations=..., debug=...)
    """
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="empty_question")

    session_id = (req.session_id or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="missing_session_id")

    # Ensure a session state exists
    st = get_state(session_id) or {}
    if not st:
        set_state(session_id, created_at=_now_iso(), summary="", turns=[])

    # Record the user turn
    append_turn(session_id, role="user", content=q)

    # Build a minimal structured context (the generator tolerates empty sections)
    summary_txt = (get_state(session_id) or {}).get("summary", "")
    last_turns = recent_turns(session_id, n=6)  # short history for coherence

    context_block = (
        "[Context]\n"
        f"- Summary: {summary_txt or 'None'}\n"
        f"- Recent turns:\n" + "\n".join(
            f"  - {t.get('role','?')}: {t.get('content','')}" for t in last_turns
        ) + "\n"
        "[End Context]\n"
    )

    # Ask the generator
    try:
        gen = generate_answer(
            question=f"{q}\n\n{context_block}",
            k=RETRIEVAL_TOP_K,
            max_context_chars=3000,
            debug=False,
            show_citations=True,
        )
    except Exception as e:
        # Record the failure briefly to help debugging, but don't crash the API schema
        append_turn(session_id, role="assistant", content=f"(internal error: {e})")
        raise HTTPException(status_code=500, detail=f"generation_failed: {type(e).__name__}: {e}")

    answer_text = (gen.get("answer") or "").strip()
    citations = gen.get("citations") or None
    dbg = gen.get("debug") or None

    # Record assistant turn & update a simple rolling summary
    append_turn(session_id, role="assistant", content=answer_text)
    try:
        # naive summary update (safe no-op if your summarizer is a stub)
        update_summary(session_id, f"Q: {q}\nA: {answer_text}")
    except Exception:
        pass

    # IMPORTANT: return the fields your ChatResponse expects (no rogue 'content' keys)
    return ChatResponse(answer=answer_text, citations=citations, debug=dbg)