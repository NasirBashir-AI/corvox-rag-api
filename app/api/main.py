# app/api/main.py
"""
FastAPI entrypoint for Corah – Version: Smart Conversational Flow
- Fully context-aware
- No repeated follow-ups
- Smooth lead capture
- Polite warm closures
"""

from __future__ import annotations
from typing import List
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import ChatRequest, ChatResponse, HealthResponse, SearchHit, SearchResponse
from app.core.config import RETRIEVAL_TOP_K
from app.core.utils import normalize_ws
from app.retrieval.retriever import search, get_facts
from app.generation.generator import generate_answer
from app.core.session_mem import (
    get_state, set_state, append_turn, recent_turns,
    update_summary, get_lead_slots, all_lead_info_complete, mark_closed
)
from app.api.intents import detect_intent, smalltalk_reply
from app.lead.capture import update_lead_info, next_lead_question

# -----------------------------------------------------------
# App setup
# -----------------------------------------------------------
app = FastAPI(title="Corah API", version="2.0.0", docs_url="/docs", redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# -----------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True)

# -----------------------------------------------------------
@app.get("/api/search", response_model=SearchResponse)
def api_search(q: str = Query(..., min_length=1), k: int = Query(RETRIEVAL_TOP_K, ge=1, le=20)) -> SearchResponse:
    try:
        raw = search(q, k=k)
        hits = [SearchHit(**{k: h.get(k) for k in SearchHit.__fields__}) for h in raw]
        return SearchResponse(hits=hits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search_failed: {e}")

# -----------------------------------------------------------
@app.get("/api/ping")
def api_ping(session_id: str = Query(..., min_length=1)) -> dict:
    st = get_state(session_id) or {}
    set_state(session_id, **{**st, "last_ping": _now_iso()})
    return {"ok": True}

# -----------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    q_raw = (req.question or "").strip()
    if not q_raw:
        raise HTTPException(status_code=400, detail="empty_question")

    session_id = (req.session_id or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="missing_session_id")

    q = normalize_ws(q_raw)

    # --- Ensure state exists ---
    st = get_state(session_id) or {"created_at": _now_iso(), "summary": "", "turns": []}
    set_state(session_id, **st)
    append_turn(session_id, "user", q)

    # --- Detect intent ---
    kind, topic = detect_intent(q)

    # --- Handle smalltalk immediately ---
    if kind == "smalltalk":
        answer = smalltalk_reply(q)
        append_turn(session_id, "assistant", answer)
        update_summary(session_id)
        return ChatResponse(answer=answer)

    # --- Lead extraction & session info ---
    slots = update_lead_info(session_id, q)
    turns = recent_turns(session_id, 6)
    facts = get_facts(["contact_email", "office_address", "contact_url"])

    # --- Conversational follow-up gating ---
    turn_idx = len(turns)
    follow_up = None
    if not all_lead_info_complete(session_id):
        follow_up = next_lead_question(session_id, turn_idx, user_intent=kind)

    # --- Build context for generator ---
    summary = get_state(session_id).get("summary", "")
    user_details = "\n".join(f"{k}: {v}" for k, v in slots.items() if v)
    contact_ctx = "\n".join(f"{k}: {v}" for k, v in facts.items() if v)
    current_topic = st.get("current_topic", topic or "general")

    context_block = (
        "[Context]\n"
        f"- Summary: {summary or 'None'}\n"
        f"- Current topic: {current_topic}\n"
        f"- Recent turns:\n" + "\n".join(
            f"  - {t['role']}: {t['content']}" for t in turns
        ) + "\n"
        f"- User details:\n{user_details or 'None'}\n"
        f"- Company contact:\n{contact_ctx or 'None'}\n"
        "[Intent]\n"
        f"kind: {kind}\n"
        f"topic: {topic or 'None'}\n"
        "[End Context]"
    )

    # --- Generate main answer ---
    gen = generate_answer(
        question=f"{q}\n\n{context_block}",
        k=RETRIEVAL_TOP_K, max_context_chars=3000,
        debug=False, show_citations=True,
    )

    answer_text = (gen.get("answer") or "").strip()
    append_turn(session_id, "assistant", answer_text)
    update_summary(session_id)

    # --- Merge polite follow-up if needed ---
    if follow_up and kind in {"lead", "contact", "pricing"}:
        answer_text += f"\n{follow_up}"

    # --- Graceful closure check ---
    if q.lower() in {"bye", "thank you", "thanks", "ok bye", "goodbye"}:
        mark_closed(session_id)
        answer_text = (
            "Thank you for your time! I hope I was helpful. "
            "I’ll close this chat now. If you’d like to continue later, just start a new chat anytime."
        )
        append_turn(session_id, "assistant", answer_text)

    return ChatResponse(answer=answer_text, citations=gen.get("citations"), debug=gen.get("debug"))