from __future__ import annotations

import os, json, re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from app.core.utils import normalize_ws

# retrieval surface
from app.retrieval.retriever import search

_PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_FINAL_MODEL   = os.getenv("OPENAI_FINAL_MODEL",   os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

_LEGACY_TEMP   = float(os.getenv("TEMPERATURE", "0.5"))
_PLANNER_TEMP  = float(os.getenv("PLANNER_TEMPERATURE", os.getenv("PLANNER_TEMP", "0.3")))
_FINAL_TEMP    = float(os.getenv("FINAL_TEMPERATURE",   os.getenv("FINAL_TEMP",   str(_LEGACY_TEMP))))

_client = OpenAI()

_CTX_START = "[Context]"
_CTX_END   = "[End Context]"

_FACT_QUERIES = [
    "address","where are you based","where are you located","location",
    "services","what services","pricing","price","cost","how much",
    "contact","email","phone","number","website","url"
]

def _split_user_and_ctx(q: str) -> Tuple[str, str]:
    if _CTX_START in q and _CTX_END in q:
        head, rest = q.split(_CTX_START, 1)
        ctx, _ = rest.split(_CTX_END, 1)
        return head.strip(), ctx.strip()
    return q.strip(), ""

def _extract_section(ctx_block: str, header: str) -> str:
    if not ctx_block: return ""
    start_pat = re.compile(rf"^\s*-\s*{re.escape(header)}\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
    m = start_pat.search(ctx_block)
    if not m: return ""
    start_idx = m.end(0)
    tail = ctx_block[start_idx:]
    nxt = re.search(r"^\s*-\s*[A-Za-z].*?:", tail, re.MULTILINE)
    chunk = tail[:nxt.start()] if nxt else tail
    text = (m.group(1) + "\n" + chunk).strip()
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()

def _extract_intent(ctx_block: str) -> Tuple[str, Optional[str], bool]:
    if not ctx_block:
        return "other", None, True
    m_kind = re.search(r"^\s*kind\s*:\s*([A-Za-z_]+)\s*$", ctx_block, re.IGNORECASE | re.MULTILINE)
    m_topic = re.search(r"^\s*topic\s*:\s*([A-Za-z_]+)\s*$", ctx_block, re.IGNORECASE | re.MULTILINE)
    m_policy = re.search(r"policy\.answer_only_until_intent:\s*(true|false)", ctx_block, re.IGNORECASE)
    kind = (m_kind.group(1).strip().lower() if m_kind else "other")
    topic = (m_topic.group(1).strip().lower() if m_topic else None)
    if topic in ("none",""): topic = None
    answer_only_until_intent = True if not m_policy else (m_policy.group(1).lower() == "true")
    return kind, topic, answer_only_until_intent

def _chat(model: str, system: str, user: str, temperature: float) -> str:
    resp = _client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()

def _planner(user_text: str, intent_kind: str, intent_topic: Optional[str]) -> Dict[str, Any]:
    needs = intent_kind in ("info","contact") or (intent_kind == "info" and intent_topic in ("pricing","services"))
    system = (
        "Return ONLY JSON: {needs_retrieval:boolean, search_query:string|null}\n"
        "- If the user asks for company facts (address/location/services/pricing/contact), needs_retrieval=true.\n"
        "- Otherwise false. search_query is usually the user text."
    )
    user = f"User text:\n{user_text}\n\nHeuristic needs_retrieval (pre): {str(needs).lower()}"
    raw = _chat(_PLANNER_MODEL, system, user, temperature=_PLANNER_TEMP)
    out = {"needs_retrieval": needs, "search_query": user_text}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            out["needs_retrieval"] = bool(obj.get("needs_retrieval", needs))
            out["search_query"] = (obj.get("search_query") or user_text)
    except Exception:
        pass
    return out

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
    answer_only_until_intent: bool,
) -> str:
    system = (
        "You are Corah, Corvox’s professional front-desk assistant.\n"
        "Rules:\n"
        "- Answer FIRST, concisely, using [Retrieved]/[Company contact]/[Pricing]—no fluff.\n"
        "- Do NOT ask any follow-up question unless intent is lead/pricing/contact, or the user explicitly asks to book/quote.\n"
        "- Never repeat the same ask; if a field exists in [User details], don’t ask again.\n"
        "- Public contact details (email, office address, website URL) may be shared verbatim if present in [Retrieved] or [Company contact].\n"
        "- Pricing is scope-based: short discovery call → tailored quote. No made-up numbers.\n"
        "- If out-of-scope, briefly redirect to Corvox services (no question).\n"
        "- Warm, business-casual tone; 1–3 sentences.\n"
    )

    shots = (
        "Ex1\nUser: where are you based?\n[Retrieved] Suite 303, Quantrill House, 2 Dunstable Road, Luton, LU1 1DX\n"
        "Assistant: We’re based at Suite 303, Quantrill House, 2 Dunstable Road, Luton, LU1 1DX.\n\n"
        "Ex2\nUser: what services do you provide?\n[Retrieved] Custom chat & voice agents for support, FAQs, guided sales.\n"
        "Assistant: We build custom AI chat and voice agents for support, FAQs, and guided sales.\n\n"
        "Ex3\nUser: pricing?\n[Pricing] Depends on scope; discovery call then clear quote.\n"
        "Assistant: Pricing depends on scope—we start with a short discovery call, then provide a clear quote.\n\n"
        "Ex4\nUser: can you book a call?\n[User details] name: Nasir; email: nasir@x.com\n"
        "Assistant: I can line that up—Friday 11am works. I’ll pass your details to the team and confirm by email.\n\n"
    )

    intent_line = f"kind={intent_kind or 'other'}; topic={intent_topic or 'None'}; answer_only_until_intent={str(answer_only_until_intent).lower()}"

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
        "Reply as Corah now."
    )
    return _chat(model, system, user, temperature=_FINAL_TEMP)

def generate_answer(
    question: str,
    k: int = 5,
    max_context_chars: int = 3000,
    debug: Optional[bool] = False,
    show_citations: Optional[bool] = False,
) -> Dict[str, Any]:
    user_text, ctx_block = _split_user_and_ctx(question)

    summary       = _extract_section(ctx_block, "Summary")
    current_topic = _extract_section(ctx_block, "Current topic")
    recent_turns  = _extract_section(ctx_block, "Recent turns")
    user_details  = _extract_section(ctx_block, "User details")
    contact_ctx   = _extract_section(ctx_block, "Company contact")
    pricing_ctx   = _extract_section(ctx_block, "Pricing")
    intent_kind, intent_topic, answer_only_until_intent = _extract_intent(ctx_block)

    plan = _planner(user_text, intent_kind=intent_kind, intent_topic=intent_topic)
    needs_retrieval = bool(plan.get("needs_retrieval", False))
    search_query = (plan.get("search_query") or user_text).strip()

    low = user_text.lower()
    if any(term in low for term in _FACT_QUERIES):
        needs_retrieval = True

    hits: List[Dict[str, Any]] = []
    retrieved_snippets = ""
    if needs_retrieval:
        try:
            raw_hits = search(search_query, k=k)
            pieces: List[str] = []
            used = 0
            for h in raw_hits:
                snippet = normalize_ws(h.get("content", "")).strip()
                if not snippet:
                    continue
                title = (h.get("title") or "")[:120]
                one = f"[{title}] {snippet}"
                if used + len(one) > max_context_chars:
                    if not pieces:
                        pieces.append(one[:max_context_chars-1] + "…")
                    break
                pieces.append(one)
                used += len(one)
            hits = raw_hits
            retrieved_snippets = "\n\n".join(pieces)
        except Exception:
            hits, retrieved_snippets = [], ""

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
        answer_only_until_intent=answer_only_until_intent,
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

    dbg = {"intent": {"kind": intent_kind, "topic": intent_topic},
           "needs_retrieval": needs_retrieval,
           "num_hits": len(hits)} if debug else None

    return {"answer": answer, "citations": citations or None, "debug": dbg}