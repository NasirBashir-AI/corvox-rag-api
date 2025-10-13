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
_TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.5"))

_client = OpenAI()

_CTX_START = "[Context]"
_CTX_END   = "[End Context]"
_LEAD_HINT_RE   = re.compile(r"^Lead hint:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_LAST_ASKED_RE  = re.compile(r"^last_asked:\s*([a-z_]+)\s*$", re.IGNORECASE | re.MULTILINE)

def _split_user_and_ctx(q: str) -> Tuple[str, str]:
    if _CTX_START in q and _CTX_END in q:
        head, rest = q.split(_CTX_START, 1)
        ctx, _ = rest.split(_CTX_END, 1)
        return head.strip(), ctx.strip()
    return q.strip(), ""

def _extract_section(ctx_block: str, header: str) -> str:
    """
    Extracts a bullet-style section that starts with "- <Header>:" and
    continues until the next "- <Other>:" line or end of block.
    """
    if not ctx_block:
        return ""
    pat = re.compile(rf"-\s*{re.escape(header)}\s*:\s*(.*)", re.IGNORECASE)
    m = pat.search(ctx_block)
    if not m:
        return ""
    start = m.end(0)
    tail = ctx_block[start:]
    nxt = re.search(r"^\s*-\s*[A-Za-z].*?:", tail, re.MULTILINE)
    chunk = tail[: nxt.start()] if nxt else tail
    text = (m.group(1) + "\n" + chunk).strip()
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()

def _extract_ctx_lines(ctx_block: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    out["lead_hint"]   = (_LEAD_HINT_RE.search(ctx_block or "") or (lambda:None)()) and _LEAD_HINT_RE.search(ctx_block or "").group(1).strip() if ctx_block else ""
    out["last_asked"]  = (_LAST_ASKED_RE.search(ctx_block or "") or (lambda:None)()) and _LAST_ASKED_RE.search(ctx_block or "").group(1).strip() if ctx_block else ""
    return out

def _chat(model: str, system: str, user: str, temperature: float = _TEMPERATURE) -> str:
    resp = _client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    return (resp.choices[0].message.content or "").strip()

def _planner(user_text: str, summary: str, current_topic: str, lead_hint: str = "") -> Dict[str, Any]:
    """
    Lightweight planner:
    - Detects simple follow-ups like "yes / sure / tell me more / go ahead" -> follow_up_on_current_topic = true
    - Avoids changing topic if the user didn't ask to
    - Keeps retrieval off for follow-ups unless the user asks for factual specifics that require docs
    """
    system = (
        "You are a tiny planner for Corvox's assistant.\n"
        "Return ONLY JSON with keys: "
        "{kind:'qa'|'lead'|'contact'|'pricing'|'other', "
        " needs_retrieval:boolean, search_query:string|null, "
        " follow_up_on_current_topic:boolean}\n"
        "Treat back-channel acknowledgements like 'yes', 'sure', 'tell me more', 'go ahead', 'sounds good' "
        "as follow_up_on_current_topic=true. Do not switch topics on those.\n"
        "Keep needs_retrieval=false unless the user explicitly requests details that need docs."
    )
    user = (
        f"User: {user_text}\n\n"
        f"Summary: {summary or '-'}\n"
        f"Current topic: {current_topic or '-'}\n"
        f"Lead hint: {lead_hint or 'none'}\n"
    )
    raw = _chat(_PLANNER_MODEL, system, user, temperature=0.0)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    # Safe default
    return {"kind": "qa", "needs_retrieval": False, "search_query": None, "follow_up_on_current_topic": False}

def _final_answer(
    model: str,
    user_text: str,
    retrieved_snippets: str,
    user_details: str,
    contact_ctx: str,
    pricing_ctx: str,
    lead_hint: str,
    last_asked: str,
    summary_ctx: str,
    current_topic_ctx: str,
    recent_turns_ctx: str,
) -> str:
    # ---- System rules (tight scope + humane style) ----
    system = (
        "You are Corah, Corvox’s warm front-desk assistant—polite, concise, genuinely helpful.\n"
        "Hard rules:\n"
        "• Scope: You are NOT a general web assistant. Stay within Corvox, our services, pricing model, lead capture, and logistics. "
        "  If asked for general trivia (e.g., celebrities, universities, weather), decline briefly and refocus on how Corvox can help.\n"
        "• Use memory: Rely on [Summary], [Current topic], and [Recent turns] to continue the SAME thread. "
        "  Never ask “what topic?” after a back-channel like “yes/sure/tell me more”.\n"
        "• Tone: 1–3 sentences, natural and friendly. Vary openers. No repeated “Hi there!”.\n"
        "• Questions: At most ONE purposeful question when it moves the goal forward. Avoid generic repeated CTAs.\n"
        "• Lead flow: If a lead hint is present, phrase exactly ONE light step for that hint (ask_<field> or bridge_back_to_<field>). "
        "  Do not nag. Respect refusals. Never restart the flow yourself.\n"
        "• Contact integrity: Use [Company contact] ONLY for Corvox’s details. "
        "  Never present anything in [User details] as company contact.\n"
        "• Don’t over-promise: Don’t say you scheduled/sent anything unless the hint indicates confirm_done.\n"
        "• Anti-repeat: If last_asked equals the current ask target, don’t repeat the same question; acknowledge and gently bridge back.\n"
    )

    # ---- Few-shot nudges for continuing the thread ----
    examples = (
        "Examples:\n"
        "User: yes, tell me more\n"
        "[Summary] topic: WhatsApp chatbot for a jewellery shop\n"
        "Assistant (good): We’d wire it to your product feed so it can answer stock and delivery instantly. "
        "Want it to read from Shopify or a Google Sheet?\n"
        "---\n"
        "User: sure\n"
        "[Summary] topic: pricing overview; user liked AI Teams tier\n"
        "Assistant (good): For AI Teams, most shops start small and scale—think foundation first, then add automations. "
        "If you share your stack (e.g., Shopify + Klaviyo), I can outline a lean starter setup.\n"
    )

    # ---- User message with structured context ----
    user = (
        f"{examples}\n"
        f"User: {user_text}\n\n"
        f"[User details]\n{user_details or 'None'}\n\n"
        f"[Company contact]\n{contact_ctx or 'None'}\n\n"
        f"[Pricing]\n{pricing_ctx or 'None'}\n\n"
        f"[Summary]\n{summary_ctx or 'None'}\n\n"
        f"[Current topic]\n{current_topic_ctx or 'None'}\n\n"
        f"[Recent turns]\n{recent_turns_ctx or 'None'}\n\n"
        f"[Retrieved]\n{retrieved_snippets or 'None'}\n\n"
        f"[Lead hint]\n{lead_hint or 'None'}\n"
        f"[last_asked]\n{last_asked or 'None'}\n\n"
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

    # structured sections from the augmented context
    user_details   = _extract_section(ctx_block, "User details")
    contact_ctx    = _extract_section(ctx_block, "Company contact")
    pricing_ctx    = _extract_section(ctx_block, "Pricing")
    summary_ctx    = _extract_section(ctx_block, "Summary")
    current_topic  = _extract_section(ctx_block, "Current topic")
    recent_turns   = _extract_section(ctx_block, "Recent turns")

    lines          = _extract_ctx_lines(ctx_block)
    lead_hint      = lines.get("lead_hint","") or ""
    last_asked     = lines.get("last_asked","") or ""

    # planner decides retrieval + follow-up behavior
    plan = _planner(user_text, summary=summary_ctx, current_topic=current_topic, lead_hint=lead_hint)
    needs_retrieval = bool(plan.get("needs_retrieval", False))
    search_query    = (plan.get("search_query") or "").strip()

    # retrieval (only when needed)
    hits: List[Dict[str, Any]] = []
    retrieved_snippets = ""
    if needs_retrieval and search_query:
        try:
            hits = search(search_query, k=k)
            pieces: List[str] = []
            total = 0
            for h in hits:
                snippet = normalize_ws(h.get("content", "")).strip()
                title = (h.get("title") or "")[:120]
                if not snippet:
                    continue
                one = f"[{title}] {snippet}"
                if total + len(one) > max_context_chars:
                    break
                pieces.append(one)
                total += len(one)
            retrieved_snippets = "\n\n".join(pieces)
        except Exception:
            hits, retrieved_snippets = [], ""

    # final answer (LLM = words only)
    answer = _final_answer(
        model=_FINAL_MODEL,
        user_text=user_text,
        retrieved_snippets=retrieved_snippets,
        user_details=user_details,
        contact_ctx=contact_ctx,
        pricing_ctx=pricing_ctx,
        lead_hint=lead_hint,
        last_asked=last_asked,
        summary_ctx=summary_ctx,
        current_topic_ctx=current_topic,
        recent_turns_ctx=recent_turns,
    ).strip()

    citations: List[Dict[str, Any]] = []
    if show_citations and hits:
        seen = set()
        for h in hits:
            key = (h.get("title"), h.get("source_uri"))
            if key in seen:
                continue
            seen.add(key)
            citations.append({
                "title": h.get("title"),
                "source_uri": h.get("source_uri"),
                "document_id": h.get("document_id"),
                "chunk_no": h.get("chunk_no"),
            })

    dbg = {"planner": plan, "used_search_query": search_query if needs_retrieval else None, "num_hits": len(hits)} if debug else None
    return {"answer": answer, "citations": citations or None, "debug": dbg}