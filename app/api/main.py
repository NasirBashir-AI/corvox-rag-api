from __future__ import annotations
from typing import List, Tuple
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import (
    ChatRequest, ChatResponse, HealthResponse, SearchHit, SearchResponse,
)
from app.core.config import RETRIEVAL_TOP_K, ANSWER_ONLY_UNTIL_INTENT_TURNS, CTA_COOLDOWN_TURNS, CTA_MAX_ATTEMPTS
from app.core.utils import normalize_ws
from app.retrieval.retriever import search
from app.generation.generator import generate_answer
from app.core.session_mem import (
    get_state, set_state, append_turn, recent_turns, update_summary,
    get_lead_slots, get_turn_count, can_offer_cta, mark_cta_used
)
from app.api.intents import detect_intent, smalltalk_reply
from app.lead.capture import update_lead_info, next_lead_question

app = FastAPI(title="Corah API", version="2.0.0",
              docs_url="/docs", redoc_url=None, openapi_url="/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def _now_iso() -> str: return datetime.now(timezone.utc).isoformat()

# --- Health ---
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True)

# --- Retrieval probe ---
@app.get("/api/search", response_model=SearchResponse)
def api_search(q: str = Query(..., min_length=1),
               k: int = Query(RETRIEVAL_TOP_K, ge=1, le=20)) -> SearchResponse:
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
            ) for h in raw
        ]
        return SearchResponse(hits=hits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search_failed: {type(e).__name__}: {e}")

# --- Ping ---
@app.get("/api/ping")
def api_ping(session_id: str = Query(..., min_length=1)) -> dict:
    st = get_state(session_id) or {}
    set_state(session_id, **{**st, "last_ping": _now_iso()})
    return {"ok": True}

# --- Chat ---
@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    q_raw = (req.question or "").strip()
    if not q_raw: raise HTTPException(status_code=400, detail="empty_question")
    session_id = (req.session_id or "").strip()
    if not session_id: raise HTTPException(status_code=400, detail="missing_session_id")
    q = normalize_ws(q_raw)

    st = get_state(session_id) or {}
    if not st:
        set_state(session_id, created_at=_now_iso(), summary="", turns=[], cta_attempts=0, cta_last_turn=0)

    append_turn(session_id, role="user", content=q)

    # 1) Intent & lead-field extraction (answer-only until intent)
    kind, topic = detect_intent(q)
    slots = update_lead_info(session_id, q)

    # 2) Smalltalk: immediate reply, no RAG
    if kind == "smalltalk":
        ans = smalltalk_reply(q)
        append_turn(session_id, role="assistant", content=ans)
        try: update_summary(session_id)
        except: pass
        return ChatResponse(answer=ans, citations=None, debug=None)

    # 3) Build context for generator
    summary_txt = (get_state(session_id) or {}).get("summary", "")
    last_turns = recent_turns(session_id, n=6)
    # User details block from slots
    user_details = "\n".join(f"{k}: {v}" for k,v in slots.items() if v)

    context_block = (
        "[Context]\n"
        f"- Summary: {summary_txt or 'None'}\n"
        f"- Current topic: {(get_state(session_id) or {}).get('current_topic','general')}\n"
        f"- Recent turns:\n" + "\n".join(f"  - {t.get('role')}: {t.get('content')}" for t in last_turns) + "\n"
        f"- User details:\n{user_details or 'None'}\n"
        "[Intent]\n"
        f"kind: {kind}\n"
        f"topic: {topic or 'None'}\n"
        "[End Context]\n"
    )

    # 4) Generate main answer (KB-first, contact-safe)
    try:
        gen = generate_answer(
            question=f"{q}\n\n{context_block}",
            k=req.k or RETRIEVAL_TOP_K,
            max_context_chars=req.max_context or 3000,
            debug=bool(req.debug),
            show_citations=req.citations,
        )
    except Exception as e:
        append_turn(session_id, role="assistant", content=f"(internal error: {e})")
        raise HTTPException(status_code=500, detail=f"generation_failed: {type(e).__name__}: {e}")

    answer_text = (gen.get("answer") or "").strip()
    citations = gen.get("citations") or None
    dbg = gen.get("debug") or None

    # 5) Answer-only until intent policy: offer lead question only when user shows intent
    turns = get_turn_count(session_id)
    shows_interest = (kind in {"lead","contact"}) or (kind=="info" and topic in {"pricing","services"})
    allow_ask = shows_interest and can_offer_cta(session_id, cooldown_turns=CTA_COOLDOWN_TURNS, max_attempts=CTA_MAX_ATTEMPTS)
    lead_q = next_lead_question(slots, allow_ask)

    if lead_q:
        answer_text = f"{answer_text}\n\n{lead_q}"
        mark_cta_used(session_id)

    append_turn(session_id, role="assistant", content=answer_text)
    try: update_summary(session_id)
    except: pass

    return ChatResponse(answer=answer_text, citations=citations, debug=dbg)