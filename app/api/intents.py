from __future__ import annotations
import re
from typing import Optional, Tuple

# Smalltalk
_SMALLTALK = [
    r"^\s*(hi|hello|hey|hiya|yo)\s*[!.]?\s*$",
    r"^\s*(good\s*(morning|afternoon|evening))\s*[!.]?\s*$",
    r"^\s*(how\s+are\s+you|how's\s+it\s+going)\s*\??\s*$",
    r"^\s*(thanks|thank\s*you)\s*[!.]?\s*$",
    r"^\s*(ok|okay|cool|great|nice)\s*[!.]?\s*$",
]

_PRICING = [r"\bprice(s|ing)?\b", r"\bcost(s)?\b", r"\brates?\b", r"\bfee(s)?\b",
            r"\bquote\b", r"\bestimate\b", r"\bper\s*month\b", r"\bsubscription\b"]
_SERVICES = [r"\bservice(s)?\b", r"\bwhat\s+do\s+you\s+do\b", r"\bcapabilit(y|ies)\b",
             r"\bsolution(s)?\b", r"\boffer(s|ings)?\b", r"\buse\s*case(s)?\b"]
_CONTACT_FOCUS = {
    "email":   r"\b(e[-\s]?mail|email|mail\s*address)\b",
    "phone":   r"\b(phone|number|call)\b",
    "address": r"\b(address|location|office|where\s+(are\s+you\s+)?(based|located))\b",
    "url":     r"\b(website|site|url)\b",
}
_LEAD = [
    r"\bbook\s*(a\s*)?call\b", r"\bcall\s*back\b", r"\bcallback\b", r"\bcontact\s*me\b",
    r"\breach\s*me\b", r"\bget\s*in\s*touch\b", r"\bdemo\b", r"\bschedule\b", r"\bquote\b",
]

_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d[\s\-()]*){7,}\d\b")

def _any(rx_list, q: str) -> bool:
    return any(re.search(p, q, re.IGNORECASE) for p in rx_list)

def _focus(q: str) -> Optional[str]:
    for k, pat in _CONTACT_FOCUS.items():
        if re.search(pat, q, re.IGNORECASE):
            return k
    return None

def detect_intent(text: str) -> Tuple[str, Optional[str]]:
    q = (text or "").strip().lower()
    if _EMAIL_RE.search(q) or _PHONE_RE.search(q):
        return "lead", None
    if any(re.search(p, q, re.IGNORECASE) for p in _SMALLTALK):
        return "smalltalk", None
    if _any(_SERVICES, q): return "info", "services"
    if _any(_PRICING, q):  return "info", "pricing"
    f = _focus(q)
    if f: return "contact", f
    if _any(_LEAD, q): return "lead", None
    return "other", None

def smalltalk_reply(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"\bhow\s+are\s+you\b", t): return "I’m doing well—thanks for asking! How can I help today?"
    if re.search(r"\bgood\s*morning\b", t):  return "Good morning! How can I help today?"
    if re.search(r"\bgood\s*afternoon\b", t):return "Good afternoon! What can I do for you?"
    if re.search(r"\bgood\s*evening\b", t):  return "Good evening! How can I help?"
    if re.search(r"\b(hi|hello|hey|hiya|yo)\b", t): return "Hi—how can I help today?"
    if re.search(r"\b(thanks|thank\s*you)\b", t):  return "You’re welcome! What can I help you with?"
    return "Hello! How can I help today?"