# app/api/main.py
"""
FastAPI entrypoint for Corah.
- /api/health  : liveness
- /api/search  : retrieval-only probe
- /api/chat    : LLM-first sandwich (planner -> retrieval -> final)
- /api/ping    : lightweight heartbeat
"""
from __future__ import annotations

from typing import List
from datetime import datetime, timezone

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
)
from app.retrieval.retriever import search
from app.generation.generator import generate_answer
# NOTE: our new capture helpers are available, but not required to boot.
# from app.lead.capture import extract_lead_fields, update_lead_info, next_lead_question

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

def _now_utc():
    return datetime.now(timezone.utc)

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="empty_question")

    # Minimal, working path: delegate to generator. (Lead-capture can be layered later.)
    try:
        result = generate_answer(
            question=q,
            k=RETRIEVAL_TOP_K,
            debug=getattr(req, "debug", False),
            show_citations=getattr(req, "show_citations", False),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generation_failed: {type(e).__name__}: {e}")

    # Build ChatResponse fields; extra fields are optional in the schema.
    return ChatResponse(
        answer=result.get("answer", ""),
        citations=result.get("citations"),
        debug=result.get("debug"),
        end_session=False,
        signals=None,
        recap=None,
    )

# ---------------------------
# Ping (optional light endpoint kept for parity)
# ---------------------------

@app.get("/api/ping")
def ping() -> dict:
    return {"ok": True, "ts": _now_utc().isoformat()}