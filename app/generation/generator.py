# app/generation/generator.py
"""
Reply generation for Corah (Phase 2 aware).

Goals
- Adapt style to sentiment (positive/neutral/frustrated) and intent (browsing→buying).
- Avoid repetition; move the chat forward with one purposeful step.
- Guard against hallucinations: do not invent prices/timelines/features.
- Keep implementation lightweight and dependency-free.

Strategy
- For informational queries: retrieve snippets and compose a concise answer.
- For buying intent: steer towards recap/lead capture (the API handles final recap).
- For frustrated tone: be brief, acknowledge, and provide a direct next step.

Inputs
- question: str
- session_id: str
- sentiment: "positive" | "neutral" | "frustrated"
- intent_level: "none" | "browsing" | "curious" | "interested" | "buying" | "explicit_cta"
- priority: "cold" | "warm" | "hot"

Returns
- reply: str
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import re

from app.retrieval.retriever import search
from app.core.session_mem import get_state
from app.core.config import HALLUCINATION_GUARD_ON


# ---------------------------
# Utilities
# ---------------------------

_PRICE_PAT = re.compile(r"(£|\$|€)\s*\d[\d,]*(\.\d{1,2})?")
_TIME_PAT = re.compile(r"\b(1|2|3|4|5|6|7|8|9|10|[1-9]\d+)\s*(day|days|week|weeks|hour|hours)\b", re.I)

def _sanitize_claims(text: str, allow_prices: bool, allow_timing: bool) -> str:
    """
    Remove risky claims if not backed by retrieved sources.
    This is a conservative guard to avoid accidental hallucinations.
    """
    out = text
    if not allow_prices:
        out = _PRICE_PAT.sub("[pricing to be confirmed]", out)
    if not allow_timing:
        out = _TIME_PAT.sub("[timeline to be confirmed]", out)
    return out


def _tone_prefix(sentiment: str) -> str:
    if sentiment == "frustrated":
        return "Sorry about that. "
    if sentiment == "positive":
        return ""
    return ""  # neutral: no prefix


def _one_next_step(intent_level: str) -> str:
    """
    Provide a single, purposeful next step (no multi-CTA spam).
    """
    if intent_level in ("buying", "explicit_cta", "interested"):
        return "If you’d like, share your email or phone so we can confirm details and move this forward."
    if intent_level in ("curious",):
        return "Would you like a quick summary of features or next steps?"
    return "Tell me a bit more about what you need, and I’ll tailor the answer."


def _compose_from_hits(hits: List[Dict[str, Any]], max_len: int = 3) -> str:
    """
    Create a short synthesis from retrieval results without over-promising.
    """
    if not hits:
        return ""
    lines = []
    for h in hits[:max_len]:
        snippet = (h.get("content") or "").strip()
        if not snippet:
            continue
        # Keep snippets short and declarative
        snippet = re.sub(r"\s+", " ", snippet)
        if len(snippet) > 280:
            snippet = snippet[:277] + "..."
        lines.append(f"• {snippet}")
    if not lines:
        return ""
    return "Here’s what I found:\n" + "\n".join(lines)


def _allow_prices_from_hits(hits: List[Dict[str, Any]]) -> bool:
    """Allow price mentions only if any snippet contains a currency symbol."""
    for h in hits or []:
        if _PRICE_PAT.search(h.get("content") or ""):
            return True
    return False


def _allow_time_from_hits(hits: List[Dict[str, Any]]) -> bool:
    """Allow timeline mentions only if any snippet contains time words."""
    for h in hits or []:
        if _TIME_PAT.search(h.get("content") or ""):
            return True
    return False


# ---------------------------
# Main entry
# ---------------------------

def generate_reply(
    question: str,
    session_id: str,
    sentiment: str = "neutral",
    intent_level: str = "none",
    priority: str = "cold",
) -> str:
    """
    Produce a safe, adaptive reply.
    """
    st = get_state(session_id) or {}
    name = st.get("name")
    company = st.get("company")
    who = company or "your business"

    # 1) Buying/Interested: steer toward closure with minimal friction.
    if intent_level in ("buying", "explicit_cta"):
        pref = _tone_prefix(sentiment)
        # No prices/timelines here; we leave specifics for proposal/recap flow
        base = f"{pref}Great — I can help set this up for {who}."
        next_step = "Please confirm your preferred contact (email or phone) and a suitable time, and I’ll arrange the follow-up."
        return f"{base} {next_step}"

    if intent_level == "interested":
        pref = _tone_prefix(sentiment)
        base = f"{pref}Got it. Based on what you’ve shared, we can tailor Corah to your needs."
        next_step = "Would you like me to summarise the options and then confirm your contact details?"
        return f"{base} {next_step}"

    # 2) Informational path: retrieve and summarise conservatively.
    try:
        hits_raw = search(question, k=5) or []
    except Exception:
        hits_raw = []

    allow_prices = _allow_prices_from_hits(hits_raw)
    allow_timing = _allow_time_from_hits(hits_raw)

    synthesis = _compose_from_hits(hits_raw, max_len=3)
    if not synthesis:
        # No evidence found – reply safely without making claims
        pref = _tone_prefix(sentiment)
        fallback = f"{pref}I couldn’t find a verified source for that yet."
        step = _one_next_step(intent_level)
        # Guard against hallucinations even in fallback
        safe = _sanitize_claims(f"{fallback} {step}", allow_prices=False, allow_timing=False)
        return safe

    # Build a short, friendly answer with one next step
    pref = _tone_prefix(sentiment)
    step = _one_next_step(intent_level)
    answer = f"{pref}{synthesis}\n\n{step}"

    # Optional hallucination guard
    if HALLUCINATION_GUARD_ON:
        answer = _sanitize_claims(answer, allow_prices=allow_prices, allow_timing=allow_timing)

    return answer