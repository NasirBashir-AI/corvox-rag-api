# app/generation/generator.py
from __future__ import annotations

import os, json, re
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from app.core.utils import normalize_ws
from app.retrieval.retriever import search, get_facts

_client = OpenAI()
client = _client

# Models & temps
_PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", "gpt-4o-mini")
_FINAL_MODEL   = os.getenv("OPENAI_FINAL_MODEL", "gpt-4o-mini")
_LEGACY_TEMP   = float(os.getenv("TEMPERATURE", "0.5"))
_PLANNER_TEMP  = float(os.getenv("PLANNER_TEMPERATURE", "0.3"))
_FINAL_TEMP    = float(os.getenv("FINAL_TEMPERATURE", str(_LEGACY_TEMP)))

# obvious “company fact” queries that should force retrieval
_FACT_QUERIES = [
    "address", "where are you based", "where are you located", "location",
    "services", "pricing", "price", "cost", "how much", "contact", "email",
    "phone", "number", "website", "url"
]

# ----------------------------
# Helpers
# ----------------------------
def _split_user_and_ctx(q: str) -> Tuple[str, str]:
    if "[Context]" in q and "[End Context]" in q:
        head, rest = q.split("[Context]", 1)
        ctx, _ = rest.split("[End Context]", 1)
        return head.strip(), ctx.strip()
    return q.strip(), ""

def _extract_section(ctx_block: str, header: str) -> str:
    if not ctx_block: return ""
    pat = re.compile(rf"-\s*{re.escape(header)}\s*:\s*(.*)", re.IGNORECASE)
    m = pat.search(ctx_block)
    if not m: return ""
    return m.group(1).strip()

def _extract_intent(ctx_block: str) -> Tuple[str, Optional[str]]:
    if not ctx_block: return "other", None
    m_kind = re.search(r"kind\s*:\s*([A-Za-z_]+)", ctx_block, re.I)
    m_topic = re.search(r"topic\s*:\s*([A-Za-z_]+)", ctx_block, re.I)
    kind = (m_kind.group(1).lower() if m_kind else "other")
    topic = (m_topic.group(1).lower() if m_topic else None)
    if topic in ("none", ""): topic = None
    return kind, topic

def _chat(model: str, system: str, user: str, temperature: float) -> str:
    resp = _client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}]
    )
    return (resp.choices[0].message.content or "").strip()

# ----------------------------
# Planner (decides if retrieval needed)
# ----------------------------
def _planner(user_text: str, intent_kind: str, intent_topic: Optional[str]) -> Dict[str, Any]:
    needs = intent_kind in ("info", "contact") or (intent_kind == "info" and intent_topic in ("pricing", "services"))
    system = (
        "Return ONLY JSON: {needs_retrieval:boolean, search_query:string|null}.\n"
        "True if user asks company facts (address, services, pricing, contact)."
    )
    user = f"User text:\n{user_text}\nPre-eval: {needs}"
    try:
        raw = _chat(_PLANNER_MODEL, system, user, _PLANNER_TEMP)
        obj = json.loads(raw)
        return {"needs_retrieval": bool(obj.get("needs_retrieval", needs)),
                "search_query": (obj.get("search_query") or user_text)}
    except Exception:
        return {"needs_retrieval": needs, "search_query": user_text}

# ----------------------------
# Final Answer Composer
# ----------------------------
def _final_answer(model: str,
                  user_text: str,
                  retrieved_snippets: str,
                  summary: str,
                  current_topic: str,
                  recent_turns: str,
                  user_details: str,
                  contact_ctx: str,
                  pricing_ctx: str,
                  intent_kind: str,
                  intent_topic: Optional[str]) -> str:
    system = (
        "You are Corah, Corvox’s professional front-desk assistant — factual, concise, warm.\n"
        "Rules:\n"
        "1) Answer directly using [Retrieved]/[Company contact] if available.\n"
        "2) If user details (name/company) exist, use them naturally once.\n"
        "3) Never repeat info already given in chat.\n"
        "4) If info missing, say so once and offer a next step, not apology loops.\n"
        "5) If intent=info, answer factually then softly offer help.\n"
        "6) If intent=lead, help progress politely (ask missing details once).\n"
        "7) If intent=contact, share public email/address/URL verbatim if in [Retrieved]/[Company contact].\n"
        "8) If question off-topic (singing, cooking etc.), redirect politely: "
        "“I focus on Corvox AI and digital services — would you like a quick overview or pricing info?”\n"
        "9) For goodbye/thanks, close warmly: “Thank you for your time. I’ll close the chat now, "
        "but you can start a new one anytime.”\n"
    )

    user = (
        f"[Intent] kind={intent_kind}; topic={intent_topic or 'None'}\n\n"
        f"User: {user_text}\n\n"
        f"[Summary]\n{summary or 'None'}\n\n"
        f"[User details]\n{user_details or 'None'}\n\n"
        f"[Company contact]\n{contact_ctx or 'None'}\n\n"
        f"[Pricing]\n{pricing_ctx or 'None'}\n\n"
        f"[Retrieved]\n{retrieved_snippets or 'None'}\n\n"
        "Reply as Corah following the rules above."
    )
    return _chat(model, system, user, _FINAL_TEMP)

# ----------------------------
# Main generator
# ----------------------------
def generate_answer(question: str,
                    k: int = 5,
                    max_context_chars: int = 3000,
                    debug: Optional[bool] = False,
                    show_citations: Optional[bool] = False) -> Dict[str, Any]:
    user_text, ctx_block = _split_user_and_ctx(question)
    summary = _extract_section(ctx_block, "Summary")
    current_topic = _extract_section(ctx_block, "Current topic")
    recent_turns = _extract_section(ctx_block, "Recent turns")
    user_details = _extract_section(ctx_block, "User details")
    contact_ctx = _extract_section(ctx_block, "Company contact")
    pricing_ctx = _extract_section(ctx_block, "Pricing")
    intent_kind, intent_topic = _extract_intent(ctx_block)

    # Planner
    plan = _planner(user_text, intent_kind, intent_topic)
    needs_retrieval = bool(plan.get("needs_retrieval"))
    search_query = (plan.get("search_query") or user_text).strip()
    if any(t in user_text.lower() for t in _FACT_QUERIES): needs_retrieval = True

    # Retrieval
    hits, retrieved_snippets = [], ""
    if needs_retrieval:
        try:
            raw_hits = search(search_query, k=k)
            snippets, total = [], 0
            for h in raw_hits:
                text = normalize_ws(h.get("content", ""))[:500]
                title = h.get("title") or ""
                one = f"[{title}] {text}"
                if total + len(one) > max_context_chars: break
                snippets.append(one); total += len(one)
            hits, retrieved_snippets = raw_hits, "\n\n".join(snippets)
        except Exception:
            pass

    # Final compose
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

    # Citations
    citations = None
    if show_citations and hits:
        citations = [{"title": h.get("title"), "source_uri": h.get("source_uri")} for h in hits]

    dbg = {"intent": intent_kind, "topic": intent_topic,
           "retrieved": len(hits)} if debug else None

    return {"answer": answer, "citations": citations, "debug": dbg}