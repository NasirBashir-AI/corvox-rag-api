# app/api/main.py
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
from app.lead.capture import (
    update_lead_info,
    next_lead_question,
)
from app.core.session_mem import (
    get_state,
    set_state,
    append_turn,
    recent_turns,
    update_summary,
    get_lead_slots,
)

app = FastAPI(title="Corah API", version="1.0.0", docs_url="/docs", redoc_url=None, openapi_url="/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True)


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="empty_question")

    session_id = req.session_id
    state = get_state(session_id) or {}
    turn_idx = int(state.get("turn_idx", 0))

    # 1) store user turn
    append_turn(session_id, role="user", content=q)
    turn_idx += 1
    set_state(session_id, turn_idx=turn_idx, last_user_ts=_utc_now())

    # 2) update lead info (extract & persist name/email/phone/company/time if present)
    slots = update_lead_info(session_id, q)  # <- ensures name sticks when the user says it

    # 3) summarise & context
    last = recent_turns(session_id, n=8)
    update_summary(session_id, "\n".join(f"{t['role']}: {t['content']}" for t in last))

    # 4) controller: if we should ask ONE lead question, do so (but only for missing fields)
    ask = next_lead_question(
        session_id=session_id,
        turn_idx=turn_idx,
        user_intent=state.get("user_intent", ""),
    )
    if ask:
        append_turn(session_id, role="assistant", content=ask)
        return ChatResponse(answer=ask)

    # 5) let the generator compose the helpful answer with KB + slots (don’t re-ask name if slots['name'])
    context_block = (
        "[Context]\n"
        f"- Summary: {state.get('summary','')}\n"
        f"- Lead slots: {slots}\n"
        f"- Recent: " + " | ".join(f"{t['role']}: {t['content']}" for t in last) + "\n"
        "[End Context]"
    )

    try:
        gen = generate_answer(f"{q}\n\n{context_block}", k=RETRIEVAL_TOP_K)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generation_failed: {type(e).__name__}: {e}")

    answer = gen["answer"].strip()
    append_turn(session_id, role="assistant", content=answer)
    return ChatResponse(answer=answer, citations=gen.get("citations"))