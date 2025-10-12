from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from app.core.utils import normalize_ws
from app.retrieval.retriever import search
from app.core.config import TEMPERATURE as CONFIG_TEMPERATURE

_PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_FINAL_MODEL   = os.getenv("OPENAI_FINAL_MODEL",   os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_TEMPERATURE   = CONFIG_TEMPERATURE  # single source of truth

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
    out["summary"] = _extract_section(ctx, "Summary")
    out["current_topic"] = _extract_section(ctx, "Current topic")
    out["recent_turns"] = _extract_section(ctx, "Recent turns")
    out["user_details"] = _extract_section(ctx, "User details")
    out["company_contact"] = _extract_section(ctx, "Company contact")
    out["pricing"] = _extract_section(ctx, "Pricing")
    out["last_asked"] = ""
    m = re.search(r"last_asked\s*:\s*([A-Za-z_]+)", ctx or "", re.IGNORECASE)
    if m: out["last_asked"] = m.group(1).strip().lower()
    return out

def _chat(model: str, system: str, user: str, temperature: float = _TEMPERATURE) -> str:
    resp = _client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    return (resp.choices[0].message.content or "").strip()

# ---------- Planner with local backchannel fast-path ----------
_BACKCHANNELS = {
    "yes", "yeah", "yep", "sure", "go ahead", "please continue",
    "tell me more", "sounds good", "okay", "ok", "alright", "cool",
    "carry on", "continue", "yup"
}

def _looks_like_backchannel(s: str) -> bool:
    t = (s or "").strip().lower()
    if t in _BACKCHANNELS:
        return True
    for pat in (r"\btell me more\b", r"\bgo ahead\b", r"\bplease continue\b", r"\bcontinue\b", r"\bsure\b", r"\byes\b", r"\bsounds good\b"):
        if re.search(pat, t):
            return True
    return False

def _planner(user_text: str, lead_hint: str = "") -> Dict[str, Any]:
    # Fast path: if the user says "yes/sure/tell me more", follow up on the current topic
    if _looks_like_backchannel(user_text):
        return {"kind": "follow_up_on_current_topic", "needs_retrieval": False, "search_query": None, "lead_prompt": None}

    system = (
        "You are a classifier/planner for a business assistant. "
        "Return ONLY JSON: {kind:'smalltalk'|'lead'|'contact'|'pricing'|'qa'|'other', "
        "needs_retrieval:boolean, search_query:string|null, lead_prompt:string|null}. "
        "Use 'lead' when user asks to start/call-back, gives phone/email, or asks how to start."
    )
    user = f"User text:\n{user_text}\n\nLead hint (optional): {lead_hint or 'none'}"
    raw = _chat(_PLANNER_MODEL, system, user, temperature=0.0)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
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
    summary: str,
    current_topic: str,
    recent_turns: str,
    planner_kind: str,
) -> str:
    system = (
        "You are Corah, Corvox’s warm front-desk assistant—polite, concise, genuinely helpful.\n"
        "Core behavior (obey strictly):\n"
        "1) Keep answers short (1–3 sentences by default), natural, and non-repetitive; vary openers.\n"
        "2) Ask at most ONE short question in a turn. No stacked CTAs.\n"
        "3) Use [Summary], [Current topic], and [Recent turns] to continue the same thread; "
        "   do NOT re-ask what the user already told you.\n"
        "4) If a lead hint is present, ask exactly ONE next step and avoid repeating the same ask if last_asked matches.\n"
        "5) Use [Company contact] ONLY for Corvox details. Never treat [User details] as company info.\n"
        "6) Respect refusals (e.g., no contact info): help anyway and offer alternatives—don’t nag.\n"
        "7) Never restart the lead flow yourself; only phrase the next step indicated.\n"
        "\n"
        "Few-shot examples:\n"
        "User: yes, tell me more\n"
        "[Summary] topic: WhatsApp chatbot for jewellery; goal: capture leads\n"
        "Assistant: Absolutely—here’s how it would plug in: we add a WhatsApp entry point, route common questions, and push interested chats into your CRM. Would you like a quick flow outline?\n"
        "\n"
        "User: sure, go ahead\n"
        "[Summary] topic: reconciliation bot for an accounting firm\n"
        "Assistant: Great—concretely, it syncs invoices, matches payments, flags discrepancies, and posts notes back to your ledger. Want me to list the data sources we’d connect?\n"
    )
    user = (
        f"User: {user_text}\n\n"
        f"[Summary]\n{summary or 'None'}\n\n"
        f"[Current topic]\n{current_topic or 'None'}\n\n"
        f"[Recent turns]\n{recent_turns or 'None'}\n\n"
        f"[User details]\n{user_details or 'None'}\n\n"
        f"[Company contact]\n{contact_ctx or 'None'}\n\n"
        f"[Pricing]\n{pricing_ctx or 'None'}\n\n"
        f"[Retrieved]\n{retrieved_snippets or 'None'}\n\n"
        f"[Lead hint]\n{lead_hint or 'None'}\n\n"
        f"[Planner]\nkind={planner_kind or 'qa'}\n\n"
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
    summary       = ctx.get("summary","")
    current_topic = ctx.get("current_topic","")
    recent_turns  = ctx.get("recent_turns","")
    user_details  = ctx.get("user_details","")
    contact_ctx   = ctx.get("company_contact","")
    pricing_ctx   = ctx.get("pricing","")
    lead_hint     = ctx.get("lead_hint","")

    # Planner (with backchannel follow-up)
    plan = _planner(user_text, lead_hint=lead_hint)
    needs_retrieval = bool(plan.get("needs_retrieval", True))
    search_query = (plan.get("search_query") or user_text).strip()
    planner_kind = plan.get("kind") or "qa"

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
        summary=summary,
        current_topic=current_topic,
        recent_turns=recent_turns,
        planner_kind=planner_kind,
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