from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI

from app.core.utils import normalize_ws
from app.retrieval.retriever import search
from app.core.config import TEMPERATURE as CONFIG_TEMPERATURE

_client = OpenAI()

_CTX_START = "[Context]"
_CTX_END   = "[End Context]"
_LEAD_HINT_RE   = re.compile(r"^Lead hint:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_TOPIC_HINT_RE  = re.compile(r"^Topic hint:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_LAST_ASKED_RE  = re.compile(r"^last_asked:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

_PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_FINAL_MODEL   = os.getenv("OPENAI_FINAL_MODEL",   os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_TEMPERATURE   = CONFIG_TEMPERATURE  # keep centralized

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

def _extract_hints(ctx: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not ctx: return out
    m1 = _LEAD_HINT_RE.search(ctx)
    m2 = _TOPIC_HINT_RE.search(ctx)
    m3 = _LAST_ASKED_RE.search(ctx)
    if m1: out["lead_hint"] = m1.group(1).strip()
    if m2: out["topic_hint"] = m2.group(1).strip()
    if m3: out["last_asked"] = m3.group(1).strip()
    return out

def _chat(model: str, system: str, user: str, temperature: float = _TEMPERATURE) -> str:
    resp = _client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    return (resp.choices[0].message.content or "").strip()

# ---------- planner: tiny router that honours topic hint on ack turns ----------
def _planner(user_text: str, topic_hint: str) -> Dict[str, Any]:
    system = (
        "You are a small planner. Return ONLY JSON with keys:\n"
        "{kind:'qa'|'lead'|'contact'|'pricing'|'smalltalk', needs_retrieval:boolean, search_query:string}\n"
        "- If user asks to arrange/book a call or provides contact -> kind:'lead'.\n"
        "- If user gives a generic acknowledgement like 'yes/sure/tell me more', prefer topic_hint for search.\n"
        "- Otherwise use the user_text as the search query.\n"
    )
    user = f"user_text: {user_text}\n\ntopic_hint: {topic_hint or 'none'}"
    raw = _chat(_PLANNER_MODEL, system, user, temperature=0.0)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    # safe fallback
    search_query = topic_hint or user_text
    return {"kind": "qa", "needs_retrieval": True, "search_query": search_query}

# ---------- final answer ----------
def _final_answer(
    user_text: str,
    retrieved_snippets: str,
    user_details: str,
    contact_ctx: str,
    pricing_ctx: str,
    lead_hint: str,
    last_asked: str,
) -> str:
    system = (
        "You are Corah, Corvox’s warm front-desk assistant—polite, concise, genuinely helpful.\n"
        "Style rules (obey strictly):\n"
        "• 1–3 sentences by default; vary your openers (no repeated 'Hi there!').\n"
        "• Ask at most ONE short question per turn; no stacked CTAs.\n"
        "• If a Lead hint is present, phrase exactly the next missing step. "
        "  If last_asked equals the same ask target, do not repeat—briefly acknowledge and gently bridge back.\n"
        "• Use [Company contact] ONLY for Corvox info. Never claim [User details] as company contact.\n"
        "• Respect refusals to share contact; keep helping and offer alternatives without nagging.\n"
        "• Keep answers on the current topic; do not drift to unrelated pricing/overview unless asked.\n"
    )
    lead_line = f"[Lead hint]\n{lead_hint}\n" if lead_hint else ""
    last_line = f"[last_asked]\n{last_asked}\n" if last_asked else ""
    user = (
        f"User: {user_text}\n\n"
        f"[User details]\n{user_details or 'None'}\n\n"
        f"[Company contact]\n{contact_ctx or 'None'}\n\n"
        f"[Pricing]\n{pricing_ctx or 'None'}\n\n"
        f"[Retrieved]\n{retrieved_snippets or 'None'}\n\n"
        f"{lead_line}{last_line}"
        "Now reply as Corah."
    )
    return _chat(_FINAL_MODEL, system, user, temperature=_TEMPERATURE)

def generate_answer(
    question: str,
    k: int = 5,
    max_context_chars: int = 3000,
    debug: Optional[bool] = False,
    show_citations: Optional[bool] = False,
) -> Dict[str, Any]:
    user_text, ctx_block = _split_user_and_ctx(question)
    sections = _extract_hints(ctx_block)
    topic_hint  = sections.get("topic_hint", "")
    lead_hint   = sections.get("lead_hint", "")
    last_asked  = sections.get("last_asked", "")

    # structured sections
    user_details = _extract_section(ctx_block, "User details")
    contact_ctx  = _extract_section(ctx_block, "Company contact")
    pricing_ctx  = _extract_section(ctx_block, "Pricing")

    plan = _planner(user_text, topic_hint=topic_hint)
    needs_retrieval = bool(plan.get("needs_retrieval", True))
    search_query = (plan.get("search_query") or topic_hint or user_text).strip()

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
        user_text=user_text,
        retrieved_snippets=retrieved_snippets,
        user_details=user_details,
        contact_ctx=contact_ctx,
        pricing_ctx=pricing_ctx,
        lead_hint=lead_hint,
        last_asked=last_asked,
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