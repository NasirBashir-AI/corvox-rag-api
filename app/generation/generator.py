# app/generation/generator.py
from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from app.core.utils import normalize_ws
from app.retrieval.retriever import search

# ----------------------------
# Models & Temperatures
# ----------------------------
_PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_FINAL_MODEL   = os.getenv("OPENAI_FINAL_MODEL",   os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

# Phase 2: separate planner/final temps; fall back to legacy TEMPERATURE
_LEGACY_TEMP   = float(os.getenv("TEMPERATURE", "0.5"))
_PLANNER_TEMP  = float(os.getenv("PLANNER_TEMPERATURE", os.getenv("PLANNER_TEMP", "0.3")))
_FINAL_TEMP    = float(os.getenv("FINAL_TEMPERATURE",   os.getenv("FINAL_TEMP",   str(_LEGACY_TEMP))))

_client = OpenAI()
# Export for lead_intent.py compatibility
client = _client

# ----------------------------
# Context markers & regex
# ----------------------------
_CTX_START = "[Context]"
_CTX_END   = "[End Context]"
_LEAD_HINT_RE = re.compile(r"^Lead hint:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_LAST_ASKED_RE = re.compile(r"^last_asked\s*:\s*([A-Za-z_]+)\s*$", re.IGNORECASE | re.MULTILINE)

# Terms that should always trigger retrieval (company facts)
_FACT_QUERIES = [
    "address", "where are you based", "where are you located", "location",
    "services", "what services", "pricing", "price", "cost", "how much",
    "contact", "email", "phone", "number"
]


# ----------------------------
# Helpers
# ----------------------------
def _split_user_and_ctx(q: str) -> Tuple[str, str]:
    if _CTX_START in q and _CTX_END in q:
        head, rest = q.split(_CTX_START, 1)
        ctx, _ = rest.split(_CTX_END, 1)
        return head.strip(), ctx.strip()
    return q.strip(), ""

def _extract_section(ctx_block: str, header: str) -> str:
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

def _extract_ctx_bits(ctx: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    out["lead_hint"] = ""
    if ctx:
        m1 = _LEAD_HINT_RE.search(ctx)
        if m1:
            out["lead_hint"] = m1.group(1).strip()
    m2 = _LAST_ASKED_RE.search(ctx or "")
    out["last_asked"] = m2.group(1).strip() if m2 else ""
    return out

def _chat(model: str, system: str, user: str, temperature: float) -> str:
    resp = _client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


# ----------------------------
# Planner (tight retrieval routing)
# ----------------------------
def _planner(user_text: str, lead_hint: str = "", current_topic: str = "") -> Dict[str, Any]:
    """
    Classifier/planner. Keeps the thread on track and forces KB retrieval for
    company facts (address/location/services/pricing/contact).
    """
    system = (
        "You are a tiny planner. Return ONLY JSON:\n"
        "{kind:'qa'|'lead'|'contact'|'pricing'|'follow_up_on_current_topic'|'out_of_scope',"
        " needs_retrieval:boolean, search_query:string|null, lead_prompt:string|null}\n"
        "- If the user asks about company facts (address, where based/location, services, pricing/cost/how much, or contact),\n"
        "  set kind:'qa', needs_retrieval:true, and use the user text as search_query.\n"
        "- If the user expresses sales intent ('can you build/make', 'interested', 'demo', 'quote', 'trial'),\n"
        "  or provides phone/email/time, classify as 'lead'.\n"
        "- Map 'yes/sure/okay/tell me more/go ahead/sounds good' to follow_up_on_current_topic.\n"
        "- Use 'out_of_scope' for unrelated trivia.\n"
        "- Set needs_retrieval true only when factual details from KB are needed."
    )
    user = (
        f"User text:\n{user_text}\n\n"
        f"Lead hint (optional): {lead_hint or 'none'}\n"
        f"Current topic (if any): {current_topic or 'none'}\n"
        "Back-channel words to treat as follow-up: yes, sure, ok, okay, go ahead, tell me more, sounds good."
    )

    raw = _chat(_PLANNER_MODEL, system, user, temperature=_PLANNER_TEMP)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    # Default fallback: basic QA with retrieval
    return {"kind": "qa", "needs_retrieval": True, "search_query": user_text, "lead_prompt": None}


# ----------------------------
# Final answer composer (help-first + no-invention)
# ----------------------------
def _final_answer(
    model: str,
    user_text: str,
    retrieved_snippets: str,
    user_details: str,
    contact_ctx: str,
    pricing_ctx: str,
    summary: str,
    current_topic: str,
    recent_turns: str,
    lead_hint: str,
    last_asked: str,
) -> str:
    # Help-first, concise, KB-first, and no invention of company facts
    system = (
        "You are Corah, Corvox’s warm front-desk assistant—helpful first, concise, factual.\n"
        "Rules:\n"
        "- Always answer the user's question before any lead capture.\n"
        "- Ask at most one short question, only if it clearly advances the goal.\n"
        "- Never ask for a field that already appears in [User details].\n"
        "- NEVER invent company facts (address, where based, services, pricing, or contact). "
        "Use [Retrieved], [Company contact], or [Pricing]; if not present, say you don’t have it in view and offer next steps.\n"
        "- Respect 'no' (or refusal to share contact) and continue helping without nagging.\n"
        "- If last_asked equals what you’re about to ask, don’t repeat—move on.\n"
        "- Pricing: give a safe, short statement (depends on scope; discovery call then quote). No made-up numbers.\n"
        "- Brand: Corvox BUILDS custom chat + voice agents (Corah). Do NOT imply we only integrate third-party tools.\n"
    )

    # Very small, concrete few-shot
    shots = (
        "Example 1\n"
        "User: where are you based?\n"
        "[Retrieved] Luton, UK\n"
        "Assistant: We’re based in Luton, UK. If you’d like more details, I can share contact info or next steps.\n\n"
        "Example 2\n"
        "User: what services do you provide?\n"
        "[Retrieved] Custom chat and voice agents for support, FAQs, product guidance.\n"
        "Assistant: We build custom chat and voice agents—covering support, FAQs, and guided recommendations. "
        "If you’d like, I can outline an approach for your use case.\n\n"
        "Example 3\n"
        "User: what's your address?\n"
        "[Retrieved] (empty)\n"
        "Assistant: I don’t have our address in view here. I can share it by email or connect you with the team—would you like that?\n\n"
        "Example 4\n"
        "User: how much does it cost?\n"
        "[Pricing] Overview: Pricing depends on scope; short discovery call, then a clear quote.\n"
        "Assistant: Pricing depends on scope. We usually start with a short discovery call and then share a clear quote. "
        "If helpful, I can note a contact so the team can follow up.\n\n"
    )

    user = (
        f"{shots}"
        f"User: {user_text}\n\n"
        f"[Summary]\n{summary or 'None'}\n\n"
        f"[Current topic]\n{current_topic or 'None'}\n\n"
        f"[Recent turns]\n{recent_turns or 'None'}\n\n"
        f"[User details]\n{user_details or 'None'}\n\n"
        f"[Company contact]\n{contact_ctx or 'None'}\n\n"
        f"[Pricing]\n{pricing_ctx or 'None'}\n\n"
        f"[Retrieved]\n{retrieved_snippets or 'None'}\n\n"
        f"[Lead hint]\n{lead_hint or 'None'}\n\n"
        f"[last_asked]\n{last_asked or 'None'}\n\n"
        "Now reply as Corah following the rules above."
    )
    return _chat(model, system, user, temperature=_FINAL_TEMP)


# ----------------------------
# Public: generate_answer
# ----------------------------
def generate_answer(
    question: str,
    k: int = 5,
    max_context_chars: int = 3000,
    debug: Optional[bool] = False,
    show_citations: Optional[bool] = False,
) -> Dict[str, Any]:
    """
    Sandwich:
      1) Planner (route + decide retrieval)
      2) Retrieval (bounded)
      3) Final compose
    """
    user_text, ctx_block = _split_user_and_ctx(question)

    # Pull structured sections the orchestrator sends
    summary       = _extract_section(ctx_block, "Summary")
    current_topic = _extract_section(ctx_block, "Current topic")
    recent_turns  = _extract_section(ctx_block, "Recent turns")
    user_details  = _extract_section(ctx_block, "User details")
    contact_ctx   = _extract_section(ctx_block, "Company contact")
    pricing_ctx   = _extract_section(ctx_block, "Pricing")

    bits          = _extract_ctx_bits(ctx_block)
    lead_hint     = bits.get("lead_hint", "")
    last_asked    = bits.get("last_asked", "")

    # Planner
    plan = _planner(user_text, lead_hint=lead_hint, current_topic=current_topic)
    kind = plan.get("kind", "qa")
    needs_retrieval = bool(plan.get("needs_retrieval", True))

    # If user text obviously contains a fact query, force retrieval (defensive)
    low = user_text.lower()
    if any(term in low for term in _FACT_QUERIES):
        needs_retrieval = True

    # Retrieval (bounded)
    hits: List[Dict[str, Any]] = []
    retrieved_snippets = ""
    if needs_retrieval and kind != "out_of_scope":
        try:
            search_query = (plan.get("search_query") or user_text).strip()
            raw_hits = search(search_query, k=k)
            pieces: List[str] = []
            total = 0
            for h in raw_hits:
                snippet = normalize_ws(h.get("content", "")).strip()
                title = (h.get("title") or "")[:120]
                if not snippet:
                    continue
                one = f"[{title}] {snippet}"
                if total + len(one) > max_context_chars:
                    break
                pieces.append(one)
                total += len(one)
            hits = raw_hits
            retrieved_snippets = "\n\n".join(pieces)
        except Exception:
            hits, retrieved_snippets = [], ""

    # Final answer
    answer = _final_answer(
        model=_FINAL_MODEL,
        user_text=user_text,
        retrieved_snippets=retrieved_snippets,
        user_details=user_details,
        contact_ctx=contact_ctx,
        pricing_ctx=pricing_ctx,
        summary=summary,
        current_topic=current_topic,
        recent_turns=recent_turns,
        lead_hint=lead_hint,
        last_asked=last_asked,
    ).strip()

    # Citations (optional)
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

    dbg = {"planner": plan, "num_hits": len(hits)} if debug else None
    return {"answer": answer, "citations": citations or None, "debug": dbg}