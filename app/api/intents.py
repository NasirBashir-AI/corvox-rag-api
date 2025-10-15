# app/api/intents.py
"""
Lightweight intent router for Corah (smalltalk & housekeeping only).

Purpose
- Quickly handle harmless, non-business intents (greetings, thanks, pleasantries)
  so the main chat flow can stay focused on lead handling and RAG.

Contract
- route_intent(text) -> dict | None
  Returns a small dict with:
    { "kind": "smalltalk", "reply": "<one-liner>" }
  or None if the text should continue through normal flow.

Notes
- Keep this conservative; do not guess business intents here.
- Pricing/quotes/booking should NOT be handled here (handled in main flow).
"""

from __future__ import annotations
from typing import Optional, Dict

def _lower(s: str) -> str:
    return (s or "").strip().lower()

_GREET = (
    "hi", "hello", "hey", "salaam", "assalam", "assalamu", "good morning", "good afternoon",
    "good evening", "hiya"
)
_THANKS = ("thanks", "thank you", "appreciate it", "cheers", "much obliged")
_POLITE = ("please", "that helps", "nice", "great", "awesome", "perfect", "cool")
_HOW_ARE_YOU = ("how are you", "how’s it going", "hows it going", "how are u")
_WHO_ARE_YOU = ("who are you", "what are you", "what is corah", "who is corah")
_RESET = ("start over", "reset chat", "new chat")

def route_intent(text: str) -> Optional[Dict[str, str]]:
    t = _lower(text)

    # Greetings
    if any(t == g or t.startswith(g + " ") for g in _GREET):
        return {
            "kind": "smalltalk",
            "reply": "Hello! I’m Corah. Tell me a bit about your business and what you’d like the chatbot to do."
        }

    # How-are-you
    if any(p in t for p in _HOW_ARE_YOU):
        return {
            "kind": "smalltalk",
            "reply": "I’m here and ready to help with Corvox services and pricing. What would you like to know?"
        }

    # Who-are-you
    if any(p in t for p in _WHO_ARE_YOU):
        return {
            "kind": "smalltalk",
            "reply": "I’m Corah, Corvox’s assistant. I can explain features, capture your details, and arrange a callback or proposal."
        }

    # Thanks (do NOT end here; main flow handles the final-check logic)
    if any(k in t for k in _THANKS):
        return {
            "kind": "smalltalk",
            "reply": "You’re welcome! Is there anything else I can help with?"
        }

    # General polite filler
    if any(k in t for k in _POLITE):
        return {
            "kind": "smalltalk",
            "reply": "Glad to help. Would you like to explore features or discuss next steps?"
        }

    # Reset/new chat hint (front-end should actually do the reset)
    if any(k in t for k in _RESET):
        return {
            "kind": "smalltalk",
            "reply": "To start fresh, click ‘Start New Chat’. Meanwhile, tell me what you’d like the chatbot to achieve."
        }

    # Otherwise let main flow handle it
    return None