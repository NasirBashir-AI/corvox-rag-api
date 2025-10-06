"""
app/api/intents.py

Lightweight intent routing for Corah.
- Never calls the retriever or LLM here; this module only classifies.
- Categories: smalltalk | contact | pricing | services | other
- If smalltalk, we also return a ready-made polite reply (no RAG needed).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from app.core.utils import normalize_ws


# -----------------------------
# Keyword lexicons (compact)
# -----------------------------

_SMALLTALK_PATTERNS = [
    r"^\s*(hi|hello|hey|yo|hiya)\s*[!.]?\s*$",
    r"^\s*(good\s*(morning|afternoon|evening))\s*[!.]?\s*$",
    r"^\s*(how\s+are\s+you|how's\s+it\s+going|sup|what'?s\s+up)\s*\??\s*$",
    r"^\s*(thanks|thank\s+you)\s*[!.]?\s*$",
    r"^\s*(ok|okay|cool|great|nice)\s*[!.]?\s*$",
]

_CONTACT_PATTERNS = [
    re.compile(r"\bphone\s*(?:no\.?|number)\b", re.IGNORECASE),
    re.compile(r"\bemail\s*(?:id|address)\b", re.IGNORECASE),
    re.compile(r"\bcall\s*-?\s*back\b", re.IGNORECASE),
    re.compile(r"\bwhere\s+(?:are\s+you\s+)?based\b", re.IGNORECASE),
    re.compile(r"\bwhats?app\b", re.IGNORECASE),
]

_CONTACT_KEYWORDS = [
    "contact",
    "get in touch",
    "reach you",
    "reach out",
    "email",
    "email address",
    "email id",
    "e-mail",
    "phone",
    "phone number",
    "contact number",
    "call",
    "call back",
    "callback",
    "whatsapp",
    "address",
    "where are you based",
    "where based",
    "location",
    "located",
    "office",
    "website",
    "url"
    # consider "site" only if you want it; it’s broad
]

_PRICING_KEYWORDS = [
    "price", "pricing", "cost", "fees", "rates", "plans", "how much",
    "subscription", "quote", "estimate",  "price list", "how much is it", "per month", "budget"
]

_SERVICES_KEYWORDS = [
    "services", "what do you do", "what does corvox do", "capabilities",
    "solutions", "offerings", "products", "what you offer", "use cases", "areas you cover"
]


# Precompile simple keyword regexes for speed/clarity
def _compile_keywords(patterns):
    """
    Treat entries as regex fragments. For plain single words, add \b…\b.
    Leave phrases/regex (those containing spaces or regex tokens) as-is.
    """
    parts = []
    for p in patterns:
        # If it already looks like regex (has \, (, [, ?, |, or spaces), keep as-is
        if any(tok in p for tok in ["\\", "(", "[", "?", "|", " "]):
            parts.append(p)
        else:
            # plain word -> word boundaries
            parts.append(rf"\b{p}\b")
    return re.compile("|".join(parts), re.IGNORECASE)


_RE_CONTACT = _compile_keywords(_CONTACT_KEYWORDS)
_RE_PRICING = _compile_keywords(_PRICING_KEYWORDS)
_RE_SERVICES = _compile_keywords(_SERVICES_KEYWORDS)
_RE_SMALLTALK = [re.compile(p, re.IGNORECASE) for p in _SMALLTALK_PATTERNS]


# -----------------------------
# Public API
# -----------------------------

def detect_intent(text: str) -> Tuple[str, Optional[str]]:
    """
    Classify a user message into one of:
      - 'smalltalk'  (returns a polite reply string as second value)
      - 'contact'
      - 'pricing'
      - 'services'
      - 'other'
    """
    q = normalize_ws(text or "")

    # 1) Smalltalk first: quick win, no retrieval
    for rx in _RE_SMALLTALK:
        if rx.search(q):
            return "smalltalk", smalltalk_reply(q)

    # 2) Contact / Pricing / Services (fast keyword checks)
    if _RE_CONTACT.search(q):
        return "contact", None
    # Extra safety net: explicit regexes for common phrasings
    for rx in _CONTACT_PATTERNS:
        if rx.search(q):
            return "contact", None

    if _RE_PRICING.search(q):
        return "pricing", None
    if _RE_SERVICES.search(q):
        return "services", None

    # 3) Fallback
    return "other", None

def smalltalk_reply(text: str) -> str:
    """
    Provide a short, friendly reply for greetings/acknowledgements.
    Keep it generic and professional; do not reference internal filenames.
    """
    t = text.lower().strip()

    # Specific patterns
    if re.search(r"\bhow\s+are\s+you\b", t):
        return "I’m doing well—thanks for asking! How can I help today?"
    if re.search(r"\bgood\s*morning\b", t):
        return "Good morning! How can I help today?"
    if re.search(r"\bgood\s*afternoon\b", t):
        return "Good afternoon! What can I do for you?"
    if re.search(r"\bgood\s*evening\b", t):
        return "Good evening! How can I help?"

    # Generic greetings / acks
    if re.search(r"\b(hi|hello|hey|hiya|yo)\b", t):
        return "Hi—how can I help today?"
    if re.search(r"\b(thanks|thank\s*you)\b", t):
        return "You’re welcome! What can I help you with?"

    # Default smalltalk
    return "Hello! How can I help today?"