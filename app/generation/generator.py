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

# Separate planner/final temps; fall back to legacy TEMPERATURE
_LEGACY_TEMP   = float(os.getenv("TEMPERATURE", "0.5"))
_PLANNER_TEMP  = float(os.getenv("PLANNER_TEMPERATURE", os.getenv("PLANNER_TEMP", "0.3")))
_FINAL_TEMP    = float(os.getenv("FINAL_TEMPERATURE",   os.getenv("FINAL_TEMP",   str(_LEGACY_TEMP))))

_client = OpenAI()
client = _client  # exported compatibility

# ----------------------------
# Context markers & regex
# ----------------------------
_CTX_START = "[Context]"
_CTX_END   = "[End Context]"

# obvious “company fact” queries that should force retrieval
_FACT_QUERIES = [
    "address", "where are you based", "where are you located", "location",
    "services", "what services", "pricing", "price", "cost", "how much",
    "contact", "email", "phone", "number", "website", "url"
]

# ----------------------------
# Helpers: context parsing
# ----------------------------
def _split_user_and_ctx(q: str) -> Tuple[str, str]:
    if _CTX_START in q and _CTX_END in q:
        head, rest = q.split(_CTX_START, 1)
        ctx, _ = rest.split(_CTX_END, 1)
        return head.strip(), ctx.strip()
    return q.strip(), ""

_SEC_HDR_RE = re.compile(r"^\s*-\s*([A-Za-z][A-Za-z _-]+)\s*:\s*(.*)$", re.IGNORECASE)

def _extract_section(ctx_block: str, header: str) -> str:
    """
    Extracts a dash-led section from the [Context] block, e.g.
      - Summary: ...
      - Recent turns:
        <lines>
    Returns the raw text for that section (may span multiple lines until the next "- <Header>:").
    """
    if not ctx_block:
        return ""
    # Find the start line for this header (dash style)
    start_pat = re.compile(rf"^\s*-\s*{re.escape(header)}\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
    m = start_pat.search(ctx_block)
    if not m:
        return ""
    start_idx = m.end(0)
    tail = ctx_block[start_idx:]
    nxt = re.search(r"^\s*-\s*[A-Za-z].*?:", tail, re.MULTILINE)
    chunk = tail[:nxt.start()] if nxt else tail
    text = (m.group(1) + "\n" + chunk).strip()
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()

def _extract_intent(ctx_block: str) -> Tuple[str, Optional[str]]:
    """
    Parse the [Intent] mini-block your main.py injects:
      [Intent]
      kind: info
      topic: services
    Returns (kind, topic_or_None)
    """
    if not ctx_block:
        return "other", None
    # Look for a literal "[Intent]" section; if not present, try to find key lines anyway.
    # We’ll just scan the whole ctx_block for 'kind:' and 'topic:' lines.
    m_kind = re.search(r"^\s*kind\s*:\s*([A-Za-z_]+)\s*$", ctx_block, re.IGNORECASE | re.MULTILINE)
    m_topic = re.search(r"^\s*topic\s*:\s*([A-Za-z_]+)\s*$", ctx_block, re.IGNORECASE | re.MULTILINE)
    kind = (m_kind.group(1).strip().lower() if m_kind else "other")
    topic = (m_topic.group(1).strip().lower() if m_topic else None)
    if topic in ("none", ""):
        topic = None
    return kind, topic

def _chat(model: str, system: str, user: str, temperature: float) -> str:
    resp = _client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()

# ----------------------------
# (Optional) Planner — lightweight/defensive
# ----------------------------
def _planner(user_text: str, intent_kind: str, intent_topic: Optional[str]) -> Dict[str, Any]:
    """
    We keep a tiny planner for robustness (JSON only).
    If intent says info/contact/pricing -> retrieval = true.
    Else leave it to heuristic + explicit fact terms.
    """
    needs = intent_kind in ("info", "contact") or (intent_kind == "info" and intent_topic in ("pricing", "services"))
    system = (
        "Return ONLY JSON with keys: {needs_retrieval:boolean, search_query:string|null}\n"
        "- If the user is asking for company facts (address/location/services/pricing/contact), needs_retrieval=true.\n"
        "- Otherwise false. search_query should usually be the user text."
    )
    user = f"User text:\n{user_text}\n\nHeuristic needs_retrieval (pre): {str(needs).lower()}"
    raw = _chat(_PLANNER_MODEL, system, user, temperature=_PLANNER_TEMP)
    out = {"needs_retrieval": needs, "search_query": user_text}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            out.update({
                "needs_retrieval": bool(obj.get("needs_retrieval", needs)),
                "search_query": (obj.get("search_query") or user_text)
            })
    except Exception:
        pass
    return out

# ----------------------------
# Final answer composer (single authoritative version)
# ----------------------------
def _final_answer(
    model: str,
    user_text: str,
    retrieved_snippets: str,
    summary: str,
    current_topic: str,
    recent_turns: str,
    user_details: str,
    contact_ctx: str,
    pricing_ctx: str,
    intent_kind: str,
    intent_topic: Optional[str],
) -> str:
    system = (
        "You are Corah, Corvox’s warm front-desk assistant—help-first, concise, factual.\n"
        "Hard rules:\n"
        "1) Answer the user’s question FIRST using [Retrieved] when available.\n"
        "2) Keep replies short (1–3 sentences). Avoid filler and repetition.\n"
        "3) Never invent company facts (address/location/services/pricing/contact). "
        "   If [Retrieved]/[Company contact]/[Pricing] lacks it, say you don’t have it in view and offer a next step.\n"
        "4) Respect ‘no’ to sharing contact; don’t nag. One brief optional CTA max.\n"
        "5) Brand: Corvox BUILDS custom chat & voice agents (Corah). Do NOT imply we only integrate third-party tools.\n"
        "6) If intent kind=info (e.g., services/pricing), give the factual answer first, then at most one soft CTA.\n"
        "7) If intent kind=contact (email/phone/address/url), only share what’s present in [Retrieved]/[Company contact]. "
        "   If missing, say you can share via follow-up instead of inventing.\n"
    )

    shots = (
        "Example A\n"
        "User: where are you based?\n"
        "[Retrieved] Luton, UK\n"
        "Assistant: We’re based in Luton, UK. If you need anything else, I can point you to the contact page.\n\n"
        "Example B\n"
        "User: what services do you provide?\n"
        "[Retrieved] Custom chat & voice agents for support, FAQs, and product guidance.\n"
        "Assistant: We build custom chat and voice agents—covering support, FAQs, and guided recommendations. "
        "If helpful, I can outline an approach for your use case.\n\n"
        "Example C\n"
        "User: how much does it cost?\n"
        "[Pricing] Overview: Depends on scope; short discovery call, then a clear quote.\n"
        "Assistant: Pricing depends on scope. We usually start with a short discovery call, then share a clear quote. "
        "If helpful, I can note a contact so the team can follow up.\n\n"
        "Example D\n"
        "User: what’s your address?\n"
        "[Retrieved] (empty)\n"
        "Assistant: I don’t have our address in view here. I can share it via our contact page or by email if you’d like.\n\n"
    )

    # Pass intent explicitly (don’t rely on the model to “find it” in prose)
    intent_line = f"kind={intent_kind or 'other'}; topic={intent_topic or 'None'}"

    user = (
        f"{shots}"
        f"[Intent]\n{intent_line}\n\n"
        f"User: {user_text}\n\n"
        f"[Summary]\n{summary or 'None'}\n\n"
        f"[Current topic]\n{current_topic or 'None'}\n\n"
        f"[Recent turns]\n{recent_turns or 'None'}\n\n"
        f"[User details]\n{user_details or 'None'}\n\n"
        f"[Company contact]\n{contact_ctx or 'None'}\n\n"
        f"[Pricing]\n{pricing_ctx or 'None'}\n\n"
        f"[Retrieved]\n{retrieved_snippets or 'None'}\n\n"
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
      1) Use [Intent] from context to guide retrieval.
      2) Retrieval (bounded).
      3) Final compose (help-first, KB-first).
    """
    user_text, ctx_block = _split_user_and_ctx(question)

    # Sections provided by the orchestrator (main.py)
    summary       = _extract_section(ctx_block, "Summary")
    current_topic = _extract_section(ctx_block, "Current topic")
    recent_turns  = _extract_section(ctx_block, "Recent turns")
    user_details  = _extract_section(ctx_block, "User details")
    contact_ctx   = _extract_section(ctx_block, "Company contact")
    pricing_ctx   = _extract_section(ctx_block, "Pricing")
    intent_kind, intent_topic = _extract_intent(ctx_block)

    # Planner (light) + heuristics
    plan = _planner(user_text, intent_kind=intent_kind, intent_topic=intent_topic)
    needs_retrieval = bool(plan.get("needs_retrieval", False))
    search_query = (plan.get("search_query") or user_text).strip()

    # Force retrieval if user_text obviously asks for a fact
    low = user_text.lower()
    if any(term in low for term in _FACT_QUERIES):
        needs_retrieval = True

    # Retrieval (bounded)
    hits: List[Dict[str, Any]] = []
    retrieved_snippets = ""
    if needs_retrieval:
        try:
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
                pieces.append(one); total += len(one)
            hits = raw_hits
            retrieved_snippets = "\n\n".join(pieces)
        except Exception:
            hits, retrieved_snippets = [], ""

    # Final answer (single, authoritative function)
    answer = _final_answer(
        model=_FINAL_MODEL,
        user_text=user_text,
        retrieved_snippets=retrieved_snippets,
        summary=summary,
        current_topic=current_topic,
        recent_turns=recent_turns,
        user_details=user_details,
        contact_ctx=contact_ctx,
        pricing_ctx=pricing_ctx,
        intent_kind=intent_kind,
        intent_topic=intent_topic,
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

    dbg = {"intent": {"kind": intent_kind, "topic": intent_topic},
           "needs_retrieval": needs_retrieval,
           "num_hits": len(hits)} if debug else None

    return {"answer": answer, "citations": citations or None, "debug": dbg}