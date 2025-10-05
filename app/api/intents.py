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

_CONTACT_KEYWORDS = [
    r"\bcontact\b",
    r"\bget in touch\b",
    r"\breach (?:you|out)\b",                 # "reach you" / "reach out"
    r"\bemail(?:\s+address)?\b",             # "email" / "email address"
    r"\be-?mail(?:\s+address)?\b",           # "e-mail" / "e mail address"
    r"\bphone(?:\s+number)?\b",              # "phone" / "phone number"
    r"\bcall\b",                             # "can I call?"
    r"\bnumber\b",                           # "your number" (kept broad, but fine)
    r"\baddress\b",                          # "what is your address"
    r"\bwhere (?:are you )?based\b",         # "where are you based?"
    r"\blocat(?:ion|ed)\b",                  # "location" / "located"
    r"\boffice\b",                           # "office address"
    r"\bwebsite\b|\bsite\b|\burl\b",         # "website / url"
]

_PRICING_KEYWORDS = [
    "price", "pricing", "cost", "fees", "rates", "plans", "how much",
    "subscription", "quote", "estimate",
]

_SERVICES_KEYWORDS = [
    "services", "what do you do", "what does corvox do", "capabilities",
    "solutions", "offerings", "products", "what you offer",
]


# Precompile simple keyword regexes for speed/clarity
def _compile_keywords(words):
    # word boundaries where it makes sense; also allow phrase contains
    escaped = [re.escape(w) for w in words]
    # Use alternation; \b around single words; phrases match as substrings
    parts = []
    for w in escaped:
        if " " in w:
            parts.append(w)
        else:
            parts.append(rf"\b{w}\b")
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