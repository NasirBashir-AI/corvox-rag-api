# app/generation/generator.py
from __future__ import annotations
import os, json, re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from app.core.utils import normalize_ws
from app.retrieval.retriever import search, get_facts
from app.core.config import PLANNER_TEMPERATURE, FINAL_TEMPERATURE, ALLOW_PUBLIC_CONTACT

_PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_FINAL_MODEL   = os.getenv("OPENAI_FINAL_MODEL",   os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

_client = OpenAI()
client = _client

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
    start_idx = m.end(0); tail = ctx_block[start_idx:]
    nxt = re.search(r"^\s*-\s*[A-Za-z].*?:", tail, re.MULTILINE)
    chunk = tail[:nxt.start()] if nxt else tail
    txt = (m.group(1) + "\n" + chunk).strip()
    return "\n".join(line.rstrip() for line in txt.splitlines()).strip()

def _extract_intent(ctx_block: str) -> Tuple[str, Optional[str], bool]:
    if not ctx_block: return "other", None, False
    m_kind  = re.search(r"^\s*kind\s*:\s*([A-Za-z_]+)\s*$",   ctx_block, re.IGNORECASE | re.MULTILINE)
    m_topic = re.search(r"^\s*topic\s*:\s*([A-Za-z_]+)\s*$",  ctx_block, re.IGNORECASE | re.MULTILINE)
    m_ask   = re.search(r"^\s*ask_ok\s*:\s*(true|false)\s*$", ctx_block, re.IGNORECASE | re.MULTILINE)
    kind = (m_kind.group(1).strip().lower() if m_kind else "other")
    topic = (m_topic.group(1).strip().lower() if m_topic else None)
    ask_ok = (m_ask and m_ask.group(1).lower() == "true")
    if topic in ("none",""): topic = None
    return kind, topic, ask_ok

def _chat(model: str, system: str, user: str, temperature: float) -> str:
    resp = _client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    return (resp.choices[0].message.content or "").strip()

def _planner(user_text: str, intent_kind: str, intent_topic: Optional[str]) -> Dict[str, Any]:
    needs = intent_kind in ("info","contact") or (intent_kind=="info" and intent_topic in ("pricing","services"))
    system = (
        "Return ONLY JSON: {needs_retrieval:boolean, search_query:string|null}\n"
        "- If the user asks for company facts (address/location/services/pricing/contact), needs_retrieval=true."
    )
    user = f"User:\n{user_text}\nHeuristic needs={str(needs).lower()}"
    raw = _chat(_PLANNER_MODEL, system, user, temperature=PLANNER_TEMPERATURE)
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
    ask_ok: bool,
) -> str:
    allow_public = "YES" if ALLOW_PUBLIC_CONTACT else "NO"
    system = (
        "You are Corah, Corvox’s front-desk assistant—help-first, concise, factual.\n"
        "Rules:\n"
        "1) Answer the user’s question FIRST using [Retrieved] when available.\n"
        "2) Keep replies short (1–3 sentences). Avoid filler and repetition.\n"
        "3) Never invent company facts. If the detail isn’t present, say you don’t have it in view and offer a next step.\n"
        f"4) Contact details policy (ALLOW_PUBLIC_CONTACT={allow_public}): If a public email, office address, or website URL is explicitly present in "
        "[Retrieved] or [Company contact] and policy is YES, you MAY share it verbatim once. Otherwise don’t invent it.\n"
        "5) Ask at most ONE short follow-up question only if ask_ok=true. If ask_ok=false, do NOT ask a question.\n"
        "6) If unrelated to Corvox/AI, politely redirect back to services/pricing.\n"
        "7) Use the person’s name from [User details] naturally once if known.\n"
        "8) Never repeat the same ask in consecutive turns (assume the orchestrator passes last_asked).\n"
    )
    shots = (
        "Example (contact present)\n"
        "User: what's your email?\n"
        "[Retrieved] info@corvox.co.uk\n"
        "Assistant: Our email is info@corvox.co.uk. If helpful, I can outline next steps.\n\n"
    )
    user = (
        f"{shots}"
        f"[Intent] kind={intent_kind or 'other'}; topic={intent_topic or 'None'}; ask_ok={str(bool(ask_ok)).lower()}\n\n"
        f"User: {user_text}\n\n"
        f"[Summary]\n{summary or 'None'}\n\n"
        f"[Current topic]\n{current_topic or 'None'}\n\n"
        f"[Recent turns]\n{recent_turns or 'None'}\n\n"
        f"[User details]\n{user_details or 'None'}\n\n"
        f"[Company contact]\n{contact_ctx or 'None'}\n\n"
        f"[Pricing]\n{pricing_ctx or 'None'}\n\n"
        f"[Retrieved]\n{retrieved_snippets or 'None'}\n\n"
        "Reply as Corah following the rules above."
    )
    return _chat(model, system, user, temperature=FINAL_TEMPERATURE)

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

    intent_kind, intent_topic, ask_ok = _extract_intent(ctx_block)

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
            pieces: List[str] = []; total = 0
            for h in raw_hits:
                snippet = normalize_ws(h.get("content","")).strip()
                title = (h.get("title") or "")[:120]
                if not snippet: continue
                one = f"[{title}] {snippet}"
                if total + len(one) > max_context_chars: break
                pieces.append(one); total += len(one)
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
        ask_ok=ask_ok,
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

    dbg = {"intent": {"kind": intent_kind, "topic": intent_topic, "ask_ok": ask_ok},
           "needs_retrieval": needs_retrieval, "num_hits": len(hits)} if debug else None

    return {"answer": answer, "citations": citations or None, "debug": dbg}