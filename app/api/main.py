# app/api/main.py
"""
FastAPI entrypoint for Corah.

Endpoints
- GET  /api/health  : liveness probe
- GET  /api/search  : retrieval probe
- GET  /api/ping    : lightweight heartbeat
- POST /api/chat    : chat orchestration (records turns, calls generator)
"""
from __future__ import annotations

from typing import List, Optional, Dict
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

# Config / building blocks
from app.core.config import RETRIEVAL_TOP_K
from app.retrieval.retriever import search, get_facts
from app.generation.generator import generate_answer
from app.lead.capture import (  # opportunistic harvesting
    harvest_email,
    harvest_phone,
    harvest_name,
    harvest_time,
    harvest_company,
)
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

# CORS — relaxed for now (tighten in prod)
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
# Lightweight heartbeat (used by the web UI)
# -----------------------------------------------------------------------------

@app.get("/api/ping")
def api_ping(session_id: str = Query(..., min_length=1)) -> dict:
    # keep a timestamp so the server knows the tab is alive
    st = get_state(session_id) or {}
    set_state(session_id, **{**st, "last_ping": _now_iso()})
    return {"ok": True}

# -----------------------------------------------------------------------------
# Helpers for /api/chat
# -----------------------------------------------------------------------------

def _harvest_into_state(session_id: str, text: str) -> None:
    """Opportunistically grab contact-ish fields from the latest user message."""
    st0 = get_state(session_id) or {}
    updates: Dict[str, Optional[str]] = {}

    nm = harvest_name(text) or st0.get("name")
    if nm and nm != st0.get("name"):
        updates["name"] = nm

    comp = harvest_company(text) or st0.get("company")
    if comp and comp != st0.get("company"):
        updates["company"] = comp

    em = harvest_email(text) or st0.get("email")
    if em and em != st0.get("email"):
        updates["email"] = em

    ph = harvest_phone(text) or st0.get("phone")
    if ph and ph != st0.get("phone"):
        updates["phone"] = ph

    tpref = harvest_time(text) or st0.get("preferred_time")
    if tpref and tpref != st0.get("preferred_time"):
        updates["preferred_time"] = tpref

    if updates:
        set_state(session_id, **updates)

def _compute_last_asked(st: dict) -> Optional[str]:
    """Derive which field we asked for most recently based on timestamps in session state."""
    fields = ("name", "contact", "time", "notes")
    ts_map: Dict[str, datetime] = {}
    for f in fields:
        iso = st.get(f"asked_for_{f}_at")
        if not iso:
            continue
        try:
            ts_map[f] = datetime.fromisoformat(iso)
        except Exception:
            pass
    if not ts_map:
        return None
    return max(ts_map.items(), key=lambda kv: kv[1])[0]

def _make_user_details(st: dict) -> str:
    return (
        f"Name: {st.get('name','-')}\n"
        f"Company: {st.get('company','-')}\n"
        f"Phone: {st.get('phone','-')}\n"
        f"Email: {st.get('email','-')}\n"
        f"Preferred time: {st.get('preferred_time','-')}"
    )

def _make_contact_context(facts: dict) -> str:
    lines = []
    if facts.get("contact_email"):
        lines.append(f"Email: {facts['contact_email']}")
    if facts.get("contact_phone"):
        lines.append(f"Phone: {facts['contact_phone']}")
    if facts.get("contact_url"):
        lines.append(f"Website: {facts['contact_url']}")
    if facts.get("office_address"):
        lines.append(f"Office: {facts['office_address']}")
    return "\n".join(lines) if lines else "None"

def _make_pricing_context(facts: dict) -> str:
    out = []
    if facts.get("pricing_overview"):
        out.append(f"Overview: {facts['pricing_overview']}")
    if facts.get("pricing_bullet"):
        out.append(f"Key point: {facts['pricing_bullet']}")
    return "\n".join(out) if out else "None"

def _format_recent_turns(turns: List[dict]) -> str:
    if not turns:
        return "None"
    def fmt(t: dict) -> str:
        role = t.get("role","")
        content = (t.get("content","") or "").strip()
        return f"{role.capitalize()}: {content}"
    return "\n".join(fmt(t) for t in turns[-6:])

# -----------------------------------------------------------------------------
# Chat
# -----------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    """
    Orchestration:
      - validate
      - record user turn
      - opportunistically harvest fields into session
      - build rich context (summary, topic, recent turns, user details, facts, pricing, lead_hint/last_asked)
      - call generator
      - record assistant turn
      - update rolling summary
      - set end_session if lead just finished
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
        set_state(session_id, created_at=_now_iso(), session_summary="", turns=[])

    # Record the user turn
    append_turn(session_id, role="user", content=q)

    # Opportunistic harvest (keeps memory fresh without nagging)
    _harvest_into_state(session_id, q)

    # Refresh state after harvesting
    st_now = get_state(session_id) or {}

    # Rolling summary / (optional) current_topic maintenance
    try:
        update_summary(session_id)
    except Exception:
        pass

    summary_txt = st_now.get("session_summary") or "-"
    current_topic = st_now.get("current_topic") or "-"
    recent_block = _format_recent_turns(recent_turns(session_id, n=6) or [])

    # Facts for contact + pricing (used by generator to avoid hallucinations)
    fact_names = [
        "contact_email", "contact_phone", "contact_url", "office_address",
        "pricing_overview", "pricing_bullet"
    ]
    facts = get_facts(fact_names) or {}
    contact_ctx = _make_contact_context(facts)
    pricing_ctx = _make_pricing_context(facts)

    # User details section
    user_details = _make_user_details(st_now)

    # Lead hints & last asked (if your capture flow stamps these)
    lead_hint = st_now.get("lead_hint") or ""       # safe default if unset
    last_asked = _compute_last_asked(st_now) or ""

    # Build the rich context block the generator expects
    context_block = (
        "[Context]\n"
        f"- Summary:\n{summary_txt}\n"
        f"- Current topic:\n{current_topic}\n"
        f"- Recent turns:\n{recent_block}\n"
        f"- User details:\n{user_details}\n"
        f"- Company contact:\n{contact_ctx}\n"
        f"- Pricing:\n{pricing_ctx}\n"
        f"Lead hint: {lead_hint}\n"
        f"last_asked: {last_asked}\n"
        "[End Context]\n"
    )

    # Ask the generator
    try:
        gen = generate_answer(
            question=f"{q}\n\n{context_block}",
            k=req.k or RETRIEVAL_TOP_K,
            max_context_chars=req.max_context or 3000,
            debug=bool(req.debug),
            show_citations=bool(req.citations),
        )
    except Exception as e:
        append_turn(session_id, role="assistant", content=f"(internal error: {e})")
        raise HTTPException(status_code=500, detail=f"generation_failed: {type(e).__name__}: {e}")

    answer_text = (gen.get("answer") or "").strip()
    citations = gen.get("citations") or None
    dbg = gen.get("debug") or None

    # Record assistant turn & update summary
    append_turn(session_id, role="assistant", content=answer_text)
    try:
        update_summary(session_id)
    except Exception:
        pass

    # Close the session if the lead was just completed by capture flow
    end_session = False
    st_after = get_state(session_id) or {}
    if st_after.get("lead_just_done"):
        end_session = True
        set_state(session_id, lead_just_done=False)  # reset the one-shot flag

    return ChatResponse(
        answer=answer_text,
        citations=citations,
        debug=dbg,
        end_session=end_session,
    )