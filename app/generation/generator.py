from __future__ import annotations
import os, json, re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from app.core.config import SHOW_CITATIONS
from app.core.utils import normalize_ws
from app.retrieval.retriever import search, get_facts

_PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_FINAL_MODEL   = os.getenv("OPENAI_FINAL_MODEL",   os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_LEGACY_TEMP   = float(os.getenv("TEMPERATURE", "0.5"))
_PLANNER_TEMP  = float(os.getenv("PLANNER_TEMPERATURE", os.getenv("PLANNER_TEMP", "0.3")))
_FINAL_TEMP    = float(os.getenv("FINAL_TEMPERATURE",   os.getenv("FINAL_TEMP",   str(_LEGACY_TEMP))))

_client = OpenAI()

_CTX_START = "[Context]"; _CTX_END = "[End Context]"
_FACT_QUERIES = ["address","where are you based","where are you located","location","services",
                 "pricing","price","cost","how much","contact","email","phone","number","website","url"]

def _split_user_and_ctx(q: str) -> Tuple[str, str]:
    if _CTX_START in q and _CTX_END in q:
        head, rest = q.split(_CTX_START, 1); ctx, _ = rest.split(_CTX_END, 1)
        return head.strip(), ctx.strip()
    return q.strip(), ""

def _extract_section(ctx_block: str, header: str) -> str:
    if not ctx_block: return ""
    m = re.search(rf"^\s*-\s*{re.escape(header)}\s*:\s*(.*)$", ctx_block, re.IGNORECASE|re.MULTILINE)
    if not m: return ""
    start = m.end(0); tail = ctx_block[start:]
    nxt = re.search(r"^\s*-\s*[A-Za-z].*?:", tail, re.MULTILINE)
    chunk = tail[:nxt.start()] if nxt else tail
    text = (m.group(1) + "\n" + chunk).strip()
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()

def _extract_intent(ctx_block: str) -> Tuple[str, Optional[str]]:
    if not ctx_block: return "other", None
    m_kind = re.search(r"^\s*kind\s*:\s*([A-Za-z_]+)\s*$", ctx_block, re.I|re.M)
    m_topic= re.search(r"^\s*topic\s*:\s*([A-Za-z_]+)\s*$", ctx_block, re.I|re.M)
    kind = (m_kind.group(1).lower() if m_kind else "other")
    topic = (m_topic.group(1).lower() if m_topic else None)
    if topic in ("", "none"): topic = None
    return kind, topic

def _chat(model: str, system: str, user: str, temperature: float) -> str:
    r = _client.chat.completions.create(model=model, temperature=temperature,
        messages=[{"role":"system","content":system},{"role":"user","content":user}])
    return (r.choices[0].message.content or "").strip()

def _planner(user_text: str, intent_kind: str, intent_topic: Optional[str]) -> Dict[str, Any]:
    needs = intent_kind in ("info","contact") or (intent_kind=="info" and intent_topic in ("pricing","services"))
    system = ("Return ONLY JSON {needs_retrieval:boolean, search_query:string|null}. "
              "If the user asks company facts (address/location/services/pricing/contact), needs_retrieval=true.")
    user = f"User:\n{user_text}\nHeuristic:{str(needs).lower()}"
    raw = _chat(_PLANNER_MODEL, system, user, temperature=_PLANNER_TEMP)
    out = {"needs_retrieval": needs, "search_query": user_text}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            out["needs_retrieval"] = bool(obj.get("needs_retrieval", needs))
            out["search_query"] = (obj.get("search_query") or user_text)
    except: pass
    return out

def _final_answer(*, model: str, user_text: str, retrieved_snippets: str,
                  summary: str, current_topic: str, recent_turns: str,
                  user_details: str, contact_ctx: str, pricing_ctx: str,
                  intent_kind: str, intent_topic: Optional[str]) -> str:
    system = (
        "You are Corah, Corvox’s professional front-desk assistant—concise, factual, warm.\n"
        "Rules:\n"
        "• Always answer the question first using [Retrieved] when available.\n"
        "• Share public contact details (email, office address, website URL) verbatim if present in [Retrieved] or [Company contact].\n"
        "• If info is missing, say so briefly and offer a next step (contact email), but don’t repeat denials.\n"
        "• Use the user’s name once if present in [User details].\n"
        "• One short CTA at most, and only if the user shows interest (pricing/demo/callback). Keep responses to 1–3 sentences.\n"
        "• No marketing fluff; don’t contradict [Retrieved]."
    )
    shots = (
        "Example\n"
        "User: where are you based?\n"
        "[Retrieved] Suite 303, Quantrill House, 2 Dunstable Road, Luton, LU1 1DX\n"
        "Assistant: We’re based at Suite 303, Quantrill House, 2 Dunstable Road, Luton, LU1 1DX.\n\n"
    )
    user = (
        f"{shots}"
        f"[Intent] kind={intent_kind or 'other'}; topic={intent_topic or 'None'}\n\n"
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
    show_citations: Optional[bool] = None,
) -> Dict[str, Any]:
    user_text, ctx_block = _split_user_and_ctx(question)
    summary       = _extract_section(ctx_block, "Summary")
    current_topic = _extract_section(ctx_block, "Current topic")
    recent_turns  = _extract_section(ctx_block, "Recent turns")
    user_details  = _extract_section(ctx_block, "User details")
    # main may not supply these—fetch facts here as source of truth
    contact_ctx   = _extract_section(ctx_block, "Company contact")
    pricing_ctx   = _extract_section(ctx_block, "Pricing")
    if not contact_ctx:
        facts = get_facts(["contact_email","office_address","contact_url"])
        if facts:
            contact_ctx = "\n".join(f"{k}: {v}" for k,v in facts.items() if v)
    if not pricing_ctx:
        pricing_ctx = "Overview: Pricing depends on scope; we start with a short discovery call, then share a clear quote."

    intent_kind, intent_topic = _extract_intent(ctx_block)

    plan = _planner(user_text, intent_kind, intent_topic)
    needs_retrieval = bool(plan.get("needs_retrieval", False))
    search_query = (plan.get("search_query") or user_text).strip()
    low = user_text.lower()
    if any(term in low for term in _FACT_QUERIES): needs_retrieval = True

    hits: List[Dict[str, Any]] = []; retrieved_snippets = ""
    if needs_retrieval:
        try:
            raw_hits = search(search_query, k=k)
            pieces: List[str] = []; total = 0
            for h in raw_hits:
                snippet = normalize_ws(h.get("content",""))
                if not snippet: continue
                title = (h.get("title") or "")[:120]
                block = f"[{title}] {snippet}"
                if total + len(block) > max_context_chars: break
                pieces.append(block); total += len(block)
            hits = raw_hits; retrieved_snippets = "\n\n".join(pieces)
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
    ).strip()

    if show_citations is None: show_citations = SHOW_CITATIONS
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

    dbg = {"intent":{"kind":intent_kind,"topic":intent_topic},
           "needs_retrieval":needs_retrieval, "num_hits":len(hits)} if debug else None

    return {"answer": answer, "citations": citations or None, "debug": dbg}