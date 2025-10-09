from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from app.core.utils import normalize_ws
from app.retrieval.retriever import search

_PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_FINAL_MODEL   = os.getenv("OPENAI_FINAL_MODEL",   os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.2"))

_client = OpenAI()

_CTX_START = "[Context]"
_CTX_END   = "[End Context]"
_LEAD_HINT_RE = re.compile(r"^Lead hint:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

def _split_user_and_ctx(q: str) -> Tuple[str, str]:
    if _CTX_START in q and _CTX_END in q:
        head, rest = q.split(_CTX_START, 1)
        ctx, _ = rest.split(_CTX_END, 1)
        return head.strip(), ctx.strip()
    return q.strip(), ""

def _extract_line(ctx_block: str, key: str) -> str:
    if not ctx_block: return ""
    m = re.search(rf"{re.escape(key)}\s*:\s*(.+)", ctx_block, re.IGNORECASE)
    return (m.group(1).strip() if m else "")

def _extract_section(ctx_block: str, header: str) -> str:
    if not ctx_block: return ""
    pat = re.compile(rf"-\s*{re.escape(header)}\s*:\s*(.*)", re.IGNORECASE)
    m = pat.search(ctx_block)
    if not m: return ""
    start = m.end(0)
    tail = ctx_block[start:]
    nxt = re.search(r"^\s*-\s*[A-Za-z].*?:", tail, re.MULTILINE)
    chunk = tail[: nxt.start()] if nxt else tail
    text = (m.group(1) + "\n" + chunk).strip()
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()

def _extract_ctx_sections(ctx: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    out["lead_hint"] = (_LEAD_HINT_RE.search(ctx or "") or (lambda:None)()) and _LEAD_HINT_RE.search(ctx or "").group(1).strip() if ctx else ""
    return out

def _chat(model: str, system: str, user: str, temperature: float = _TEMPERATURE) -> str:
    resp = _client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    return (resp.choices[0].message.content or "").strip()

def _planner(user_text: str, lead_hint: str = "") -> Dict[str, Any]:
    system = (
        "You are a classifier/planner for a business assistant. "
        "Return ONLY JSON: {kind:'smalltalk'|'lead'|'contact'|'pricing'|'qa'|'other', "
        "needs_retrieval:boolean, search_query:string|null, lead_prompt:string|null} "
        "Use 'lead' when user asks to start/call-back, gives phone/email, or asks how to start."
    )
    user = f"User text:\n{user_text}\n\nLead hint (optional): {lead_hint or 'none'}"
    raw = _chat(_PLANNER_MODEL, system, user, temperature=0.0)
    try:
        return json.loads(raw) if isinstance(json.loads(raw), dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"kind": "qa", "needs_retrieval": True, "search_query": user_text, "lead_prompt": None}

def _final_answer(
    model: str,
    user_text: str,
    retrieved_snippets: str,
    user_details: str,
    contact_ctx: str,
    pricing_ctx: str,
    lead_hint: str,
) -> str:
    system = (
        "You are Corah, a professional, friendly assistant for Corvox.\n"
        "Rules (follow strictly):\n"
        "1) Keep answers short, warm, and specific (1–3 sentences unless detail was requested).\n"
        "2) Use [Company contact] ONLY for Corvox contact info. "
        "   NEVER use values from [User details] as company contact. If company phone/email is missing, say so briefly.\n"
        "3) If the user asks about their own name/phone/email, read it from [User details] and answer plainly.\n"
        "4) If a lead hint is present, ask exactly ONE short follow-up to progress the lead (no repeats).\n"
        "5) Do not dump all contact/pricing info. Provide only what was requested or what is necessary.\n"
        "6) If unsure, say so briefly and suggest the next helpful step.\n"
        "7) Do NOT start replies with greetings (“Hi/Hello”) or repeat the user’s name every turn. "
        "   Only acknowledge their name once, when you first capture it.\n"
        "8) If the user asks to arrange a call/callback, you CAN arrange it."
        "   Ask for exactly the next missing detail (name → phone/email → time), one short question at a time."
        "   Do not tell them to email unless they explicitly ask for contact information.\n"
        "9) Treat values in [User details] as the customer's contact only (for callbacks), never as Corvox’s."
    )
    user = (
        f"User: {user_text}\n\n"
        f"[User details]\n{user_details or 'None'}\n\n"
        f"[Company contact]\n{contact_ctx or 'None'}\n\n"
        f"[Pricing]\n{pricing_ctx or 'None'}\n\n"
        f"[Retrieved]\n{retrieved_snippets or 'None'}\n\n"
        f"[Lead hint]\n{lead_hint or 'None'}\n\n"
        "Now reply as Corah."
    )
    return _chat(model, system, user, temperature=_TEMPERATURE)

def generate_answer(
    question: str,
    k: int = 5,
    max_context_chars: int = 3000,
    debug: Optional[bool] = False,
    show_citations: Optional[bool] = False,
) -> Dict[str, Any]:
    user_text, ctx_block = _split_user_and_ctx(question)
    ctx = _extract_ctx_sections(ctx_block)

    # pull structured sections
    user_details  = _extract_section(ctx_block, "User details")
    contact_ctx   = _extract_section(ctx_block, "Company contact")
    pricing_ctx   = _extract_section(ctx_block, "Pricing")
    lead_hint     = ctx.get("lead_hint","")

    plan = _planner(user_text, lead_hint=lead_hint)
    needs_retrieval = bool(plan.get("needs_retrieval", True))
    search_query = (plan.get("search_query") or user_text).strip()

    hits: List[Dict[str, Any]] = []
    retrieved_snippets = ""
    if needs_retrieval:
        try:
            hits = search(search_query, k=k)
            pieces: List[str] = []
            total = 0
            for h in hits:
                snippet = normalize_ws(h.get("content", "")).strip()
                title = (h.get("title") or "")[:120]
                if not snippet: continue
                one = f"[{title}] {snippet}"
                if total + len(one) > max_context_chars: break
                pieces.append(one); total += len(one)
            retrieved_snippets = "\n\n".join(pieces)
        except Exception:
            hits, retrieved_snippets = [], ""

    answer = _final_answer(
        model=_FINAL_MODEL,
        user_text=user_text,
        retrieved_snippets=retrieved_snippets,
        user_details=user_details,
        contact_ctx=contact_ctx,
        pricing_ctx=pricing_ctx,
        lead_hint=lead_hint,
    ).strip()

    citations: List[Dict[str, Any]] = []
    if show_citations and hits:
        seen = set()
        for h in hits:
            key = (h.get("title"), h.get("source_uri"))
            if key in seen: continue
            seen.add(key)
            citations.append({
                "title": h.get("title"),
                "source_uri": h.get("source_uri"),
                "document_id": h.get("document_id"),
                "chunk_no": h.get("chunk_no"),
            })

    dbg = {"planner": plan, "used_search_query": search_query if needs_retrieval else None, "num_hits": len(hits)} if debug else None
    return {"answer": answer, "citations": citations or None, "debug": dbg}