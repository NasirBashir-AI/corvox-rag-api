"""
app/corah_ingest/extract_facts.py

Extracts structured facts (contact & pricing) from Markdown text.
Returns a list of dicts: {"name": str, "value": str, "uri": Optional[str]}

Facts we aim to capture (doc-driven only; no code constants):
- contact_email
- contact_phone
- contact_url
- office_address
- pricing_overview
- pricing_bullet (repeatable)

Notes
- Keep this robust but conservative: prefer precision over guessing.
- We only parse plaintext Markdown (no external fetches).
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional


# -----------------------------
# Regexes (conservative)
# -----------------------------

RE_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
RE_URL_MD = re.compile(r"\[[^\]]+\]\((https?://[^\s)]+)\)")  # [text](https://...)
RE_URL_BARE = re.compile(r"\bhttps?://[^\s)>\]]+\b")
RE_PHONE = re.compile(
    r"(?:\+?\d[\d\s\-\(\)]{7,}\d)"  # generic international-ish
)

RE_ADDR_LINE = re.compile(r"^\s*(address|office)\s*:\s*(.+)$", re.IGNORECASE)

# Headings for pricing/contact sections
RE_H2 = re.compile(r"^\s*#{1,3}\s*(.+?)\s*$")  # # .. ### ...
PRICING_KEYS = {"pricing", "plans", "packages"}
CONTACT_KEYS = {"contact", "get in touch", "reach us", "contact us"}


# -----------------------------
# Helpers
# -----------------------------

def _norm_line(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _looks_like_address(line: str) -> bool:
    """
    Very light heuristic: contains a number + street-ish token or a postcode-like token.
    """
    l = line.strip()
    if len(l) < 8:
        return False
    # UK-style postcodes or a comma-separated multi-part address
    if re.search(r"\b[A-Z]{1,2}\d[\dA-Z]?\s*\d[A-Z]{2}\b", l, re.IGNORECASE):
        return True
    return bool(re.search(r"\d+[, ]+\w+", l))


def _in_section(lines: List[str], start_idx: int) -> List[str]:
    """
    Collect lines under a heading until the next heading or EOF.
    """
    out: List[str] = []
    for i in range(start_idx + 1, len(lines)):
        if RE_H2.match(lines[i]):  # next heading
            break
        out.append(lines[i])
    return out


def _find_headings(lines: List[str]) -> List[tuple]:
    """
    Return list of (idx, heading_text_lower, original_text)
    """
    heads = []
    for i, ln in enumerate(lines):
        m = RE_H2.match(ln)
        if m:
            txt = m.group(1).strip()
            heads.append((i, txt.lower(), txt))
    return heads


# -----------------------------
# Extraction
# -----------------------------

def extract_facts_from_markdown(text: str, uri: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Parse a Markdown document and return structured facts.
    """
    facts: List[Dict[str, str]] = []
    if not text or not text.strip():
        return facts

    lines = text.splitlines()

    # ---- global scans (email/phone/url/address anywhere) ----
    # emails
    for m in RE_EMAIL.finditer(text):
        facts.append({"name": "contact_email", "value": m.group(0), "uri": uri or ""})
        break  # capture first; docs should have a single canonical email

    # urls: prefer Markdown links, else bare urls
    url_captured = False
    for m in RE_URL_MD.finditer(text):
        facts.append({"name": "contact_url", "value": m.group(1), "uri": uri or ""})
        url_captured = True
        break
    if not url_captured:
        m2 = RE_URL_BARE.search(text)
        if m2:
            facts.append({"name": "contact_url", "value": m2.group(0), "uri": uri or ""})

    # phone
    for m in RE_PHONE.finditer(text):
        # Clean phone: collapse spaces
        val = re.sub(r"\s+", " ", m.group(0)).strip()
        facts.append({"name": "contact_phone", "value": val, "uri": uri or ""})
        break

    # address: either explicit "Address:" / "Office:" or an address-looking line
    for ln in lines:
        m = RE_ADDR_LINE.match(ln)
        if m:
            facts.append({"name": "office_address", "value": _norm_line(m.group(2)), "uri": uri or ""})
            break
    else:
        # No explicit Address: look for a plausible address line
        for ln in lines:
            if _looks_like_address(ln):
                facts.append({"name": "office_address", "value": _norm_line(ln), "uri": uri or ""})
                break

    # ---- section-based scans (Pricing / Contact) ----
    heads = _find_headings(lines)
    for idx, lower, original in heads:
        key = lower.strip().lower()
        section = _in_section(lines, idx)

        # Pricing section
        if any(k in key for k in PRICING_KEYS):
            # overview: the first non-empty paragraph (not a bullet)
            para = []
            for ln in section:
                if ln.strip().startswith(("#", ">", "|")):
                    continue
                if ln.strip().startswith(("-", "*")):
                    break
                if ln.strip():
                    para.append(_norm_line(ln))
                elif para:
                    break
            if para:
                facts.append({"name": "pricing_overview", "value": " ".join(para).strip(), "uri": uri or ""})

            # bullets under pricing
            for ln in section:
                s = ln.strip()
                if s.startswith(("-", "*")):
                    bullet = _norm_line(s.lstrip("-*"))
                    if bullet:
                        facts.append({"name": "pricing_bullet", "value": bullet, "uri": uri or ""})

        # Contact section (in case details are grouped there)
        if any(k in key for k in CONTACT_KEYS):
            # probe section lines for explicit lines
            for ln in section:
                # Email
                em = RE_EMAIL.search(ln)
                if em:
                    facts.append({"name": "contact_email", "value": em.group(0), "uri": uri or ""})
                # Phone
                ph = RE_PHONE.search(ln)
                if ph:
                    facts.append({"name": "contact_phone", "value": _norm_line(ph.group(0)), "uri": uri or ""})
                # Address
                am = RE_ADDR_LINE.match(ln)
                if am:
                    facts.append({"name": "office_address", "value": _norm_line(am.group(2)), "uri": uri or ""})
                # URL
                um = RE_URL_MD.search(ln) or RE_URL_BARE.search(ln)
                if um:
                    val = um.group(1) if hasattr(um, "group") and um.re == RE_URL_MD else um.group(0)  # type: ignore[attr-defined]
                    if isinstance(val, tuple):
                        val = val[0]
                    facts.append({"name": "contact_url", "value": str(val), "uri": uri or ""})

    # De-dup (keep order; last one wins for singletons except pricing_bullet)
    dedup: List[Dict[str, str]] = []
    seen_singleton = {}
    for f in facts:
        name = f["name"]
        if name == "pricing_bullet":
            dedup.append(f)
            continue
        seen_singleton[name] = f  # last wins
    # rebuild: singletons first (latest), then bullets in original order
    singles = list(seen_singleton.values())
    bullets = [f for f in facts if f["name"] == "pricing_bullet"]
    return singles + bullets