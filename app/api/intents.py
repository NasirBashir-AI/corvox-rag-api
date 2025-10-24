# app/api/intents.py
"""
Lightweight intent routing.
Returns (kind, topic):
  kind  ∈ {'smalltalk','info','contact','lead','out_of_scope','other'}
  topic ∈ {'services','pricing','email','phone','address','url', None}
"""

from __future__ import annotations
import re
from typing import Optional, Tuple
from app.core.utils import normalize_ws

_SMALLTALK_PATTERNS = [
    r"^\s*(hi|hello|hey|hiya|yo)\s*[!.]?\s*$",
    r"^\s*(good\s*(morning|afternoon|evening))\s*[!.]?\s*$",
    r"^\s*(how\s+are\s+you|how's\s+it\s+going)\s*\??\s*$",
    r"^\s*(thanks|thank\s*you)\s*[!.]?\s*$",
    r"^\s*(ok|okay|cool|great|nice)\s*[!.]?\s*$",
]

_PRICING = [
    r"\bprice(s|ing)?\b", r"\bcost(s)?\b", r"\brates?\b", r"\bfee(s)?\b",
    r"\bplan(s)?\b", r"\bhow\s+much\b", r"\bquote\b", r"\bestimate\b",
    r"\bsubscription\b", r"\bbudget\b",
]
_SERVICES = [
    r"\bservice(s)?\b", r"\bwhat\s+do\s+you\s+do\b",
    r"\bcapabilit(y|ies)\b", r"\bsolution(s)?\b", r"\boffer(s|ings)?\b",
    r"\bproduct(s)?\b", r"\buse\s*case(s)?\b",
]
_LEAD = [
    r"\b(book|schedule|arrange)\s+(a\s*)?(call|callback|discovery)\b",
    r"\bcall\s*back\b", r"\bcontact\s*me\b", r"\breach\s*me\b",
    r"\bget\s+in\s+touch\b", r"\bdemo\b", r"\bconsultation\b",
]

_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d[\s\-()]*){7,}\d\b")

def _rx_any(patterns):
    rxs = [re.compile(p, re.IGNORECASE) for p in patterns]
    return lambda q: any(rx.search(q) for rx in rxs)

_is_smalltalk = lambda q: any(re.search(p, q, re.IGNORECASE) for p in _SMALLTALK_PATTERNS)
_is_pricing   = _rx_any(_PRICING)
_is_services  = _rx_any(_SERVICES)
_is_lead      = _rx_any(_LEAD)

def _contact_focus(q: str) -> Optional[str]:
    if re.search(r"\b(e[-\s]?mail|email)\b", q, re.IGNORECASE):   return "email"
    if re.search(r"\b(phone|number|call)\b", q, re.IGNORECASE):   return "phone"
    if re.search(r"\b(address|hq|office|location|where\s+are\s+you\s+(based|located))\b", q, re.IGNORECASE):
        return "address"
    if re.search(r"\b(website|site|url)\b", q, re.IGNORECASE):    return "url"
    return None

def detect_intent(text: str) -> Tuple[str, Optional[str]]:
    q = normalize_ws(text or "")

    # Provided PII → treat as lead continuation
    if _EMAIL_RE.search(q) or _PHONE_RE.search(q):
        return "lead", None

    if _is_smalltalk(q):  return "smalltalk", None
    if _is_services(q):   return "info", "services"
    if _is_pricing(q):    return "info", "pricing"

    cf = _contact_focus(q)
    if cf:                return "contact", cf
    if _is_lead(q):       return "lead", None

    # Very rough out-of-scope detection
    if re.search(r"\b(cook|recipe|sing|song|football|movie|weather)\b", q, re.IGNORECASE):
        return "out_of_scope", None
    return "other", None

def smalltalk_reply(text: str) -> str:
    t = (text or "").lower().strip()
    if re.search(r"\bhow\s+are\s+you\b", t): return "I’m doing well—thanks for asking! How can I help today?"
    if re.search(r"\bgood\s*morning\b", t):  return "Good morning! How can I help today?"
    if re.search(r"\bgood\s*afternoon\b", t):return "Good afternoon! What can I do for you?"
    if re.search(r"\bgood\s*evening\b", t):  return "Good evening! How can I help?"
    if re.search(r"\b(hi|hello|hey|hiya|yo)\b", t): return "Hi—how can I help today?"
    if re.search(r"\b(thanks|thank\s*you)\b", t):   return "You’re welcome! What can I help you with?"
    return "Hello! How can I help today?"