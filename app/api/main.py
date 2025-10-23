# app/api/main.py
from __future__ import annotations
from typing import List
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import ChatRequest, ChatResponse, HealthResponse, SearchHit, SearchResponse
from app.core.config import RETRIEVAL_TOP_K, ASK_EARLY, ASK_MIN_TURN_INDEX, ALLOW_PUBLIC_CONTACT
from app.core.utils import normalize_ws
from app.retrieval.retriever import search, get_facts
from app.generation.generator import generate_answer
from app.core.session_mem import (
    get_state, set_state, append_turn, recent_turns, update_summary,
    get_lead_slots,
)
from app.api.intents import detect_intent, smalltalk_reply

app = FastAPI(title="Corah API", version="1.0.0", docs_url="/docs", redoc_url=None, openapi_url="/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True)

@app.get("/api/search", response_model=SearchResponse)
def api_search(q: str = Query(..., min_length=1), k: int = Query(RETRIEVAL_TOP_K, ge=1, le=20)) -> SearchResponse:
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

@app.get("/api/ping")
def api_ping(session_id: str = Query(..., min_length=1)) -> dict:
    st = get_state(session_id) or {}
    set_state(session_id, **{**st, "last_ping": _now_iso()})
    return {"ok": True}

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    q_raw = (req.question or "").strip()
    if not q_raw: raise HTTPException(status_code=400, detail="empty_question")
    session_id = (req.session_id or "").strip()
    if not session_id: raise HTTPException(status_code=400, detail="missing_session_id")

    q = normalize_ws(q_raw)

    st = get_state(session_id) or {}
    if not st:
        set_state(session_id, created_at=_now_iso(), summary="", turns=[], cta_last_turn=0, cta_attempts=0)

    append_turn(session_id, role="user", content=q)

    # Intent
    kind, topic = detect_intent(q)

    # Bye / close
    if kind == "bye":
        append_turn(session_id, role="assistant", content="Understood — I’ll close this chat. If you need anything later, just start a new chat. Take care!")
        set_state(session_id, is_closed=True, closed_at=_now_iso())
        return ChatResponse(answer="Understood — I’ll close this chat. If you need anything later, just start a new chat. Take care!", end_session=True)

    # Smalltalk: answer directly (no RAG, no ask)
    if kind == "smalltalk":
        answer = smalltalk_reply(q)
        append_turn(session_id, role="assistant", content=answer)
        try: update_summary(session_id)
        except Exception: pass
        return ChatResponse(answer=answer)

    # Build [User details] from lead slots
    slots = get_lead_slots(session_id)
    user_details = "\n".join(f"{k}: {v}" for k, v in slots.items() if v)

    # Structured facts block for contact
    facts = get_facts(["contact_email","office_address","contact_url"])
    contact_ctx = "\n".join([f"{k}: {v}" for k, v in facts.items() if v])

    # Pricing context fallback (can be empty; model handles safely)
    pricing_ctx = "Overview: Pricing depends on scope; we start with a short discovery call and then provide a clear quote."

    # Decide if we allow a follow-up question THIS turn (default: later in the flow or explicit salesy intent)
    turns_count = (get_state(session_id) or {}).get("turns_count", 0)
    ask_ok = bool(
        (kind in {"lead","info","contact"} and topic in {None,"services","pricing"}) or  # salesy/info flow
        (turns_count >= ASK_MIN_TURN_INDEX)                                             # or later in convo
    )
    if not ASK_EARLY and turns_count < 2:
        ask_ok = False  # never ask on the very first replies

    # Construct context
    summary_txt = (get_state(session_id) or {}).get("summary", "")
    last_turns = recent_turns(session_id, n=6)
    context_block = (
        "[Context]\n"
        f"- Summary: {summary_txt or 'None'}\n"
        f"- Current topic: {(get_state(session_id) or {}).get('current_topic','None')}\n"
        f"- Recent turns:\n" + "\n".join(f"  - {t.get('role','?')}: {t.get('content','')}" for t in last_turns) + "\n"
        f"- User details: \n{user_details or 'None'}\n"
        f"- Company contact: \n{contact_ctx or 'None'}\n"
        f"- Pricing: \n{pricing_ctx or 'None'}\n"
        "[Intent]\n"
        f"kind: {kind}\n"
        f"topic: {topic or 'None'}\n"
        f"ask_ok: {'true' if ask_ok else 'false'}\n"
        "[End Context]\n"
    )

    # Generate
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
        # update summary/topic for follow-ups
        st2 = get_state(session_id) or {}
        st2["current_topic"] = kind if kind in {"info","contact","lead"} else st2.get("current_topic","general")
        set_state(session_id, **st2)
        update_summary(session_id)
    except Exception:
        pass

    return ChatResponse(answer=answer_text, citations=citations, debug=dbg)