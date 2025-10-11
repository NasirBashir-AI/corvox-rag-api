# app/generation/generator.py
from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from app.core.utils import normalize_ws
from app.retrieval.retriever import search

# -----------------------------
# Config
# -----------------------------

_PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_FINAL_MODEL   = os.getenv("OPENAI_FINAL_MODEL",   os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
_TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.5"))

_client = OpenAI()

# -----------------------------
# Context parsing helpers
# -----------------------------

_CTX_START = "[Context]"
_CTX_END   = "[End Context]"

# Lines we inject from main.py:
#   Lead hint: ask_name | ask_contact | ask_time | ask_notes | bridge_back_to_<field> | confirm_done | after_done
#   last_asked: name|contact|time|notes
_LEAD_HINT_RE  = re.compile(r"^Lead hint:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_LAST_ASKED_RE = re.compile(r"^last_asked:\s*(\w+)\s*$", re.IGNORECASE | re.MULTILINE)

def _split_user_and_ctx(q: str) -> Tuple[str, str]:
    if _CTX_START in q and _CTX_END in q:
        head, rest = q.split(_CTX_START, 1)
        ctx, _ = rest.split(_CTX_END, 1)
        return head.strip(), ctx.strip()
    return q.strip(), ""

def _extract_line(ctx_block: str, key: str) -> str:
    if not ctx_block:
        return ""
    m = re.search(rf"{re.escape(key)}\s*:\s*(.+)", ctx_block, re.IGNORECASE)
    return (m.group(1).strip() if m else "")

def _extract_section(ctx_block: str, header: str) -> str:
    """
    Given the raw [Context] block, pull the lines after “- <header>:”
    until the next “- <other>:” or end. Robust to minor formatting.
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

def _extract_ctx_bits(ctx: str) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"lead_hint": None, "last_asked": None}
    if not ctx:
        return out
    m1 = _LEAD_HINT_RE.search(ctx)
    if m1:
        out["lead_hint"] = (m1.group(1) or "").strip()
    m2 = _LAST_ASKED_RE.search(ctx)
    if m2:
        out["last_asked"] = (m2.group(1) or "").strip().lower()
    return out

# -----------------------------
# LLM calls
# -----------------------------

def _chat(model: str, system: str, user: str, temperature: float = _TEMPERATURE) -> str:
    try:
        resp = _client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        # Very defensive fallback to avoid 500s
        return "Sorry—I'm having trouble phrasing that right now. Could you rephrase or ask again?"

def _planner(user_text: str, lead_hint: str = "") -> Dict[str, Any]:
    """
    First-pass LLM “planner”: classify and (optionally) rewrite a search query.
    Return JSON with keys:
      - kind: 'smalltalk' | 'lead' | 'contact' | 'pricing' | 'qa' | 'other'
      - needs_retrieval: bool
      - search_query: string or null
    """
    system = (
        "You are Corah, Corvox’s friendly front-desk assistant. Voice: warm, calm, and practical. "
        "Be human, not salesy.\n"
        "\n"
        "Style & tone (obey):\n"
        "• Default to 1–3 short sentences. Ask at most ONE short question only when it moves things forward.\n"
        "• Do NOT mirror or restate the user’s intent (“It sounds like you’re interested in…”). Move to the next useful step.\n"
        "• No auto-greetings after the first turn. Use the user’s name only when you first learn it or when it’s naturally helpful.\n"
        "• Stay on topic. Tailor answers to the user’s industry and cues; avoid generic capability lists.\n"
        "• Prefer concrete, specific ideas over abstractions. (E.g., for a jewellery shop WhatsApp bot: appointment booking, repair/collection updates, new-drop alerts.)\n"
        "\n"
        "Lead flow (when hints are present):\n"
        "• If you have a lead hint, ask ONLY the next missing detail. Do not repeat the same ask in back-to-back turns. "
        "  If last_asked equals the current ask target, briefly acknowledge and gently bridge back.\n"
        "• If the hint is confirm_done, reply with a warm one-line confirmation that recaps name, phone/email, preferred time, and any note—then reassure next steps.\n"
        "• If the user declines to share contact, respect it. Offer one alternative (the company email) once, then continue helping without pushing.\n"
        "\n"
        "Grounding:\n"
        "• Use [Company contact] ONLY for Corvox details. NEVER present [User details] as company contact. If company info is missing, say so briefly.\n"
        "• When asked to “tell me more”, answer with 3–5 crisp bullets relevant to their scenario—no fluff, no boilerplate.\n"
        "• Avoid filler phrases like “it sounds like…”, “we can create custom solutions…”, or repeating what they just said.\n"
        "\n"
        "Style examples:\n"
        "- User: “What could a WhatsApp bot do for my jewellery shop?”\n"
        "  You: “A few high-impact ideas:\n"
        "  • Book viewings & repairs with reminders\n"
        "  • ‘New drop’ & back-in-stock alerts for specific collections\n"
        "  • Order/repair status updates and care tips\n"
        "  • Quick answers on metals, sizing, returns\n"
        "  Want me to sketch a quick flow?”\n"
        "\n"
        "- (Lead hint: ask=phone_or_email, last_asked=contact) User: “I’d rather not share that.”\n"
        "  You: “No problem—happy to continue here. If you ever prefer email, you can reach the team at the company address in [Company contact]. "
        "  Would you like examples of how the bot would greet customers or capture interest?”\n"
        "\n"
        "- (Lead hint: confirm_done)\n"
        "  You: “All set—Nasir, we’ll call 07922229622 on Mondays between 3–7 PM (note: WhatsApp chatbot). We’ll follow up if anything changes.”\n"
        ")"
    )
    user = f"User text:\n{user_text}\n\nLead hint (optional): {lead_hint or 'none'}"
    raw = _chat(_PLANNER_MODEL, system, user, temperature=0.0)

    # Tolerant JSON extraction
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Safe default
    return {"kind": "qa", "needs_retrieval": True, "search_query": user_text}

def _final_answer(
    model: str,
    user_text: str,
    retrieved_snippets: str,
    user_details: str,
    contact_ctx: str,
    pricing_ctx: str,
    effective_hint: Optional[str],
    last_asked: Optional[str],
) -> str:
    """
    Second LLM pass: compose the actual reply using retrieved evidence + facts,
    and a compact lead hint (if any). This model WRITES ALL WORDS; no boilerplate from code.
    """
    # Convert the hint into a small “intent” line the model can follow naturally
    hint_text = effective_hint or "none"

    system = (
        "You are Corah, Corvox’s friendly front-desk assistant—polite, extra-friendly, and genuinely helpful.\n"
        "\n"
        "Voice & Style: warm, human, and concise; use contractions; acknowledge the user’s intent; be encouraging. "
        "Default to 1–3 sentences. Vary openers (don’t say the same phrase every turn). Use the user’s name once when first captured; otherwise sparingly.\n"
        "\n"
        "Conversation rules (follow strictly):\n"
        "1) Ask at most ONE short question per turn—no stacked CTAs.\n"
        "2) If there is a Lead hint, ask only the next step it specifies. "
        "   If last_asked equals that step, do NOT repeat—briefly help/acknowledge and gently bridge back.\n"
        "3) Use [Company contact] ONLY for Corvox details; never present values in [User details] as company info. "
        "   If company info is missing, say so briefly.\n"
        "4) If the user asks for their saved details (name/phone/email), read them from [User details] plainly.\n"
        "5) Never restart or advance any lead flow on your own; phrase only the hinted next step.\n"
        "6) If the user declines to share contact, respect it; offer alternatives and keep helping.\n"
        "7) Avoid boilerplate; tailor replies to what was asked.\n"
    )
    user = (
        f"User: {user_text}\n\n"
        f"[User details]\n{user_details or 'None'}\n\n"
        f"[Company contact]\n{contact_ctx or 'None'}\n\n"
        f"[Pricing]\n{pricing_ctx or 'None'}\n\n"
        f"[Retrieved evidence]\n{retrieved_snippets or 'None'}\n\n"
        f"[Lead hint]\n{hint_text}\n\n"
        f"[last_asked]\n{last_asked or 'none'}\n\n"
        "Reply as Corah now—be specific, calm, and natural; no repetition."
    )
    return _chat(model, system, user, temperature=_TEMPERATURE)

# -----------------------------
# Public API (called by app/api/main.py)
# -----------------------------

def generate_answer(
    question: str,
    k: int = 5,
    max_context_chars: int = 3000,
    debug: Optional[bool] = False,
    show_citations: Optional[bool] = False,
) -> Dict[str, Any]:
    """
    Two-pass LLM pipeline:
      1) Planner LLM: classify + (optional) search query rewrite
      2) Retrieval (if needed)
      3) Final LLM: compose the answer with facts + evidence + compact hint
    """
    # 0) Split the augmented question: user text + [Context]
    user_text, ctx_block = _split_user_and_ctx(question)

    # Pull structured sections from context
    user_details = _extract_section(ctx_block, "User details")
    contact_ctx  = _extract_section(ctx_block, "Company contact")
    pricing_ctx  = _extract_section(ctx_block, "Pricing")

    bits = _extract_ctx_bits(ctx_block)
    lead_hint  = (bits.get("lead_hint") or "").strip() or None
    last_asked = (bits.get("last_asked") or "").strip() or None

    # If we’re being told to ask the same thing we just asked, convert to a bridge
    effective_hint = lead_hint
    if lead_hint and last_asked:
        # lead_hint patterns: ask_name / ask_contact / ask_time / ask_notes / bridge_back_to_<field> / confirm_done / after_done
        m = re.match(r"ask_(name|contact|time|notes)$", lead_hint)
        if m and m.group(1) == last_asked:
            effective_hint = f"bridge_back_to_{last_asked}"

    # 1) Planner
    plan = _planner(user_text, lead_hint=effective_hint or "")
    needs_retrieval = bool(plan.get("needs_retrieval", True))
    search_query = (plan.get("search_query") or user_text).strip()

    # 2) Retrieval (only if needed)
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
                if not snippet:
                    continue
                one = f"[{title}] {snippet}"
                if total + len(one) > max_context_chars:
                    break
                pieces.append(one)
                total += len(one)
            retrieved_snippets = "\n\n".join(pieces)
        except Exception:
            # proceed without retrieval if it fails
            hits = []
            retrieved_snippets = ""

    # 3) Final LLM
    answer = _final_answer(
        model=_FINAL_MODEL,
        user_text=user_text,
        retrieved_snippets=retrieved_snippets,
        user_details=user_details,
        contact_ctx=contact_ctx,
        pricing_ctx=pricing_ctx,
        effective_hint=effective_hint,
        last_asked=last_asked,
    ).strip()

    # Citations (keep schema-friendly: only include fields your API schema allows)
    citations: List[Dict[str, Any]] = []
    if show_citations and hits:
        seen = set()
        for h in hits:
            key = (h.get("title"), h.get("chunk_no"))
            if key in seen:
                continue
            seen.add(key)
            citations.append({
                "title": h.get("title"),
                "chunk_no": h.get("chunk_no"),
            })

    dbg = None
    if debug:
        dbg = {
            "planner": plan,
            "used_search_query": search_query if needs_retrieval else None,
            "num_hits": len(hits),
            "effective_hint": effective_hint,
            "last_asked": last_asked,
        }

    return {"answer": answer, "citations": citations or None, "debug": dbg}