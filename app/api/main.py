# app/api/main.py
from __future__ import annotations
from typing import Dict, Any, Optional, List
import os
import re
import json

import psycopg2
from fastapi import FastAPI, APIRouter, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from app.api.schemas import (
    QuestionIn, AnswerOut, SearchResponse,
    ChatRequest, ChatResponse,
    LeadStart, LeadMessageIn, LeadOut, Lead,
)
from app.retrieval.retriever import search
from app.generation.generator import generate_answer
from app.core.config import DB_URL

# -----------------------------
# FastAPI app + CORS
# -----------------------------
app = FastAPI(title="Corah API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later to your domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
api = APIRouter(prefix="/api")

# -----------------------------
# In-memory session store
# -----------------------------
SESSIONS: Dict[str, Dict[str, Any]] = {}  # { session_id: {history:[], lead:{...}} }

def session(sid: Optional[str]) -> Dict[str, Any]:
    if not sid:
        return {}
    if sid not in SESSIONS:
        SESSIONS[sid] = {"history": [], "lead": {}}
    return SESSIONS[sid]

# -----------------------------
# Health
# -----------------------------
@api.get("/health")
def health():
    # Optionally test DB connectivity quickly (commented to stay fast)
    # try:
    #     with psycopg2.connect(DB_URL) as cx, cx.cursor() as cur:
    #         cur.execute("SELECT 1")
    # except Exception as e:
    #     return {"ok": False, "db": False, "err": str(e)}
    return {"ok": True}

# -----------------------------
# Search (dev utility)
# -----------------------------
@api.get("/search", response_model=SearchResponse)
def search_endpoint(q: str = Query(..., min_length=2), k: int = 5):
    hits = search(q, k=k)
    return {"query": q, "k": k, "hits": hits}

# -----------------------------
# Answer (prod surface: answer-only)
# -----------------------------
@api.post("/answer", response_model=AnswerOut)
def answer(req: QuestionIn):
    out = generate_answer(req.question, k=req.k)
    return AnswerOut(answer=out["answer"])

# -----------------------------
# Chat (uses generator: self-query + similarity gate + temp memory)
# -----------------------------
@api.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    st = session(payload.session_id)
    if st:
        st["history"].append({"role": "user", "content": payload.question})

    out = generate_answer(
        question=payload.question,
        k=payload.k,
        max_context_chars=payload.max_context,
    )

    if st:
        st["history"].append({"role": "assistant", "content": out["answer"]})

    # We keep rewritten_query/used available (UI can ignore/hide in prod)
    return ChatResponse(
        answer=out["answer"],
        rewritten_query=out.get("rewritten_query"),
        used=out.get("used"),
    )

# =============================
# Lead Capture (natural, progressive)
# =============================

EMAIL_RX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RX = re.compile(r"^[\d+\-\s]{7,20}$")

LEAD_ORDER = [
    "name",
    "email",
    "phone",
    "company_name",
    "company_size",
    "industry",
    "requirements",
    "goals",
    "role",  # added to identify decision-maker/influencer/staff
]

PROMPTS = {
    "name": "Great—what’s your full name?",
    "email": "Thanks. What’s the best email to reach you?",
    "phone": "And a phone number (optional, but helpful)?",
    "company_name": "What’s your company name?",
    "company_size": "Roughly how many people work there?",
    "industry": "Which industry are you in?",
    "requirements": "What do you want us to build or solve for you?",
    "goals": "What’s the main goal or outcome you want to achieve?",
    "role": "What’s your role in the company (e.g., owner, manager, developer)?",
}

def valid(field: str, val: str) -> bool:
    if field == "email":
        return bool(EMAIL_RX.search(val))
    if field == "phone":
        return bool(PHONE_RX.match(val))
    return bool(val.strip())

def next_missing(lead: Dict[str, Any]) -> List[str]:
    return [f for f in LEAD_ORDER if not lead.get(f)]

# ---- Lead summary (LLM, toggleable) ----
client = OpenAI()  # uses OPENAI_API_KEY
USE_LLM_LEAD_SUMMARY = os.getenv("USE_LLM_LEAD_SUMMARY", "1") == "1"
LEAD_SUMMARY_MODEL = os.getenv("LEAD_SUMMARY_MODEL", "gpt-4o-mini")

def summarize_lead_rule_based(lead: Dict[str, Any]) -> str:
    parts = []
    if lead.get("name"): parts.append(f"Lead: {lead['name']}")
    if lead.get("company_name"): parts.append(f"Company: {lead['company_name']}")
    if lead.get("company_size"): parts.append(f"Size: {lead['company_size']}")
    if lead.get("industry"): parts.append(f"Industry: {lead['industry']}")
    if lead.get("role"): parts.append(f"Role: {lead['role']}")
    if lead.get("requirements"): parts.append(f"Requirements: {lead['requirements']}")
    if lead.get("goals"): parts.append(f"Goals: {lead['goals']}")
    return " | ".join(parts) or "Lead captured."

def summarize_lead_llm(lead: Dict[str, Any]) -> str:
    try:
        sys = (
            "You are Corah, a concise pre-sales assistant. "
            "Write a short, professional lead brief for the sales team. "
            "Use British English. Do not invent details."
        )
        user = (
            "Create a crisp lead report (4–6 lines) covering: person, company, size/industry, "
            "their role, what they want (requirements), their main goal, and a 1-line recommendation.\n\n"
            f"Lead JSON:\n{json.dumps(lead, ensure_ascii=False)}"
        )
        rsp = client.chat.completions.create(
            model=LEAD_SUMMARY_MODEL,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=220,
        )
        return rsp.choices[0].message.content.strip()
    except Exception:
        return summarize_lead_rule_based(lead)

def build_lead_summary(lead: Dict[str, Any]) -> str:
    return summarize_lead_llm(lead) if USE_LLM_LEAD_SUMMARY else summarize_lead_rule_based(lead)

# ---- DB upsert for leads ----
def upsert_lead(lead: Dict[str, Any]) -> int:
    """
    Insert a lead row. Returns lead id.
    Columns expected: name,email,phone,company_name,company_size,industry,requirements,goals,role,lead_summary
    """
    fields = [
        "name","email","phone","company_name","company_size",
        "industry","requirements","goals","role","lead_summary"
    ]
    vals = [lead.get(k) for k in fields]
    with psycopg2.connect(DB_URL) as cx, cx.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.leads
                (name,email,phone,company_name,company_size,industry,requirements,goals,role,lead_summary)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id;
            """,
            vals
        )
        lead_id = cur.fetchone()[0]
        return int(lead_id)

@api.post("/lead/start", response_model=LeadOut)
def lead_start(payload: LeadStart):
    st = session(payload.session_id)
    st["lead"] = st.get("lead", {})
    missing = next_missing(st["lead"])
    first = missing[0] if missing else None
    reply = PROMPTS.get(first, "Tell me a bit about you and your company.") if first else \
            "I already have your details. Would you like to add anything else?"
    return LeadOut(reply=reply, done=not bool(missing), missing=missing)

@api.post("/lead/message", response_model=LeadOut)
def lead_message(payload: LeadMessageIn):
    st = session(payload.session_id)
    st["lead"] = st.get("lead", {})
    user_text = payload.message.strip()

    missing = next_missing(st["lead"])
    if not missing:
        # already complete → ensure saved
        if "lead_summary" not in st["lead"]:
            st["lead"]["lead_summary"] = build_lead_summary(st["lead"])
        lead_id = upsert_lead(st["lead"])
        return LeadOut(reply="Thanks — I’ve got everything I need. Our team will reach out shortly.",
                       done=True, missing=[], lead_id=lead_id)

    field = missing[0]
    val = user_text

    if field == "email":
        m = EMAIL_RX.search(user_text)
        if not m:
            return LeadOut(reply="That doesn’t look like a valid email. Please provide something like name@company.com.",
                           done=False, missing=missing)
        val = m.group(0)
    elif field == "phone":
        if not PHONE_RX.match(user_text):
            return LeadOut(reply="That doesn’t look like a valid phone number. Include digits, spaces, + or - only.",
                           done=False, missing=missing)

    st["lead"][field] = val

    # next prompt or finish
    missing = next_missing(st["lead"])
    if not missing:
        st["lead"]["lead_summary"] = build_lead_summary(st["lead"])
        lead_id = upsert_lead(st["lead"])
        return LeadOut(reply="Perfect — captured everything. Our team will contact you soon.",
                       done=True, missing=[], lead_id=lead_id)

    nxt = missing[0]
    return LeadOut(reply=PROMPTS.get(nxt, f"Please provide your {nxt.replace('_',' ')}."),
                   done=False, missing=missing)

# Mount router
app.include_router(api)