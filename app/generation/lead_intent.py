# app/generation/lead_intent.py
"""
Lead intent classifier (lightweight, Phase 2).

Goal
- Detect buying readiness from the latest user message.
- Emit a small set of "signals" that explain the decision.

Returns
- dict with:
    {
      "interest_level": "none" | "browsing" | "curious" | "interested" | "buying" | "explicit_cta",
      "signals": [str, ...]
    }

Notes
- Pure heuristic (no external models), fast and deterministic.
- Keep it conservative; only escalate when cues are clear.
"""

from __future__ import annotations
from typing import List, Dict, Any

def _lower(s: str) -> str:
    return (s or "").strip().lower()

# Cues (ordered by strength)
_EXPLICIT_CTA = (
    "prepare a proposal", "send proposal", "book a call", "schedule a call", "call me",
    "email me", "get started", "go ahead", "let's proceed", "lets proceed", "proceed",
    "sign up", "sign-up", "subscribe", "take my details", "send me a quote", "finalise",
)
_BUYING = (
    "how much", "pricing", "price", "cost", "quote", "invoice", "trial to paid",
    "implementation timeline", "launch date", "how soon", "when can we start",
)
_INTERESTED = (
    "i want", "we want", "we need", "i need", "can you build", "can you do", "integrate",
    "integration", "connect to", "works with", "support", "features", "capabilities",
)
_CURIOUS = (
    "how does it work", "how it works", "what can it do", "tell me more",
    "explain", "overview", "details", "what is corah", "what are your services",
)
_BROWSING = (
    "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "great", "cool",
    "who are you", "how are you",
)

def _score_from_text(text: str) -> Dict[str, Any]:
    t = _lower(text)
    signals: List[str] = []

    # Explicit CTA (strongest)
    if any(p in t for p in _EXPLICIT_CTA):
        for p in _EXPLICIT_CTA:
            if p in t:
                signals.append(p)
        return {"interest_level": "explicit_cta", "signals": signals}

    # Buying
    if any(p in t for p in _BUYING):
        for p in _BUYING:
            if p in t:
                signals.append(p)
        return {"interest_level": "buying", "signals": signals}

    # Interested
    if any(p in t for p in _INTERESTED):
        for p in _INTERESTED:
            if p in t:
                signals.append(p)
        return {"interest_level": "interested", "signals": signals}

    # Curious
    if any(p in t for p in _CURIOUS):
        for p in _CURIOUS:
            if p in t:
                signals.append(p)
        return {"interest_level": "curious", "signals": signals}

    # Browsing / smalltalk
    if any(p in t for p in _BROWSING):
        for p in _BROWSING:
            if p in t:
                signals.append(p)
        return {"interest_level": "browsing", "signals": signals}

    # Default
    return {"interest_level": "none", "signals": []}


def classify_lead_intent(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    turns: list of { "role": "user"|"assistant", "content": "..." }
    We focus on the latest user message.
    """
    if not turns:
        return {"interest_level": "none", "signals": []}

    # Find last user message
    last_user = None
    for t in reversed(turns):
        if t.get("role") == "user":
            last_user = t.get("content") or ""
            break

    if not last_user:
        return {"interest_level": "none", "signals": []}

    return _score_from_text(last_user)