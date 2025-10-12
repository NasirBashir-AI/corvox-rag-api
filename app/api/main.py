"""
app/api/main.py

FastAPI entrypoint for Corah.
Pipeline: user -> planner -> retrieval -> final LLM (words only)
Controller (capture.py) owns the flow and returns only hints; LLM phrases it.
"""

from __future__ import annotations
import re
from typing import List, Optional
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SearchHit,
    SearchResponse,
)
from app.core.config import RETRIEVAL_TOP_K
from app.retrieval.retriever import search, get_facts
from app.generation.generator import generate_answer
from app.lead.capture import (
    in_progress as lead_in_progress,
    start as lead_start,
    take_turn as lead_turn,
    harvest_email,
    harvest_phone,
    harvest_name,
)
from app.core.session_mem import (
    get_state, set_state, append_turn, recent_turns,
    update_summary,
)

app = FastAPI(
    title="Corah API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -------------- health/search --------------

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

# -------------- chat --------------

_ACK_PAT = re.compile(
    r"^(yes|yep|yeah|sure|ok|okay|alright|go ahead|sounds good|great|please continue|tell me more|continue|carry on|go on)\b",
    re.I,
)

def _is_acknowledgement(text: str) -> bool:
    return bool(_ACK_PAT.search((text or "").strip()))


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="empty_question")

    session_id = req.session_id

    # record user turn and opportunistic field harvest
    append_turn(session_id, "user", q)

    st0 = get_state(session_id)
    name  = harvest_name(q)   or st0.get("name")
    phone = harvest_phone(q)  or st0.get("phone")
    email = harvest_email(q)  or st0.get("email")
    set_state(session_id, name=name, phone=phone, email=email)

    # keep rolling summary/topic
    st1 = update_summary(session_id, last_user_text=q)
    current_topic = st1.get("current_topic") or ""

    # controller hint (no copy)
    lead_signal = None
    if lead_in_progress(session_id):
        lead_signal = lead_turn(session_id, q)  # {"hint": "..."}
    else:
        # only start when user asks explicitly to arrange/ book a call OR provides contact
        ql = q.lower()
        wants_callback = any(
            kw in ql for kw in [
                "callback", "call back", "schedule a call", "book a call",
                "arrange a call", "can you call", "call me"
            ]
        )
        if wants_callback or phone or email:
            lead_signal = lead_start(session_id, kind="callback")

    hint_str = lead_signal.get("hint") if isinstance(lead_signal, dict) else None

    # compute last_asked from asked_* timestamps
    st2 = get_state(session_id)
    ts_pairs: List[tuple[str, str]] = []
    for field, key in (("name","asked_for_name_at"),
                       ("contact","asked_for_contact_at"),
                       ("time","asked_for_time_at"),
                       ("notes","asked_for_notes_at")):
        t = st2.get(key)
        if t: ts_pairs.append((field, t))
    last_asked = None
    if ts_pairs:
        try:
            last_asked = max(ts_pairs, key=lambda kv: datetime.fromisoformat(kv[1]))[0]
        except Exception:
            last_asked = ts_pairs[-1][0]

    # facts
    fact_names = ["contact_email", "contact_phone", "contact_url", "office_address",
                  "pricing_bullet", "pricing_overview"]
    facts = get_facts(fact_names) or {}
    contact_lines = []
    if facts.get("contact_email"):  contact_lines.append(f"Email: {facts['contact_email']}")
    if facts.get("contact_phone"):  contact_lines.append(f"Phone: {facts['contact_phone']}")
    if facts.get("contact_url"):    contact_lines.append(f"Website: {facts['contact_url']}")
    if facts.get("office_address"): contact_lines.append(f"Office: {facts['office_address']}")
    contact_context = "\n".join(contact_lines) if contact_lines else "None available"

    pricing_context = ""
    if facts.get("pricing_overview"):
        pricing_context += f"Overview: {facts['pricing_overview']}\n"
    if facts.get("pricing_bullet"):
        pricing_context += f"Key point: {facts['pricing_bullet']}\n"

    # user details & summary
    user_details = (
        f"Name: {name or '-'}\n"
        f"Phone: {phone or '-'}\n"
        f"Email: {email or '-'}\n"
        f"Preferred time: {st2.get('preferred_time','-')}\n"
        f"Timezone: {st2.get('timezone','-')}"
    )

    # context lines
    lines_hint = []
    if current_topic:
        lines_hint.append(f"Topic hint: {current_topic}")
    if hint_str:
        lines_hint.append(f"Lead hint: {hint_str}")
    if last_asked:
        lines_hint.append(f"last_asked: {last_asked}")
    hint_block = ("\n".join(lines_hint) + "\n") if lines_hint else ""

    # recognition: if the user said "yes/sure/tell me more", nudge planner with topic
    ack_flag = _is_acknowledgement(q)
    if ack_flag and current_topic and ("Topic hint:" not in hint_block):
        hint_block = f"Topic hint: {current_topic}\n" + hint_block

    augmented_q = (
        f"{q}\n\n"
        f"[Context]\n"
        f"- Summary:\n{st2.get('session_summary','') or 'None'}\n"
        f"- User details:\n{user_details}\n"
        f"- Company contact:\n{contact_context or 'None'}\n"
        f"- Pricing:\n{pricing_context or 'None'}\n"
        f"{hint_block}"
        f"[End Context]\n"
    )

    result = generate_answer(
        question=augmented_q,
        k=req.k or RETRIEVAL_TOP_K,
        max_context_chars=req.max_context or 3000,
        debug=req.debug,
        show_citations=req.citations,
    )

    answer = (result.get("answer") or "").strip()
    append_turn(session_id, "assistant", answer)

    end_session = False
    st_after = get_state(session_id)
    if st_after.get("lead_just_done"):
        end_session = True
        set_state(session_id, lead_just_done=False)

    return ChatResponse(
        answer=answer,
        citations=result.get("citations"),
        debug=result.get("debug"),
        end_session=end_session,
    )