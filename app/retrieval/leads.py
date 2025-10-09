# app/retrieval/leads.py
from __future__ import annotations

import uuid
from typing import Optional, Dict, Any

from app.core.utils import pg_cursor
from app.core.config import DB_URL


# ---------------------------
# DDL: structured table + safe indexes
# ---------------------------
DDL = """
CREATE SCHEMA IF NOT EXISTS corah_store;

CREATE TABLE IF NOT EXISTS corah_store.leads (
  id             BIGSERIAL PRIMARY KEY,
  session_id     TEXT NOT NULL,
  name           TEXT,
  phone          TEXT,
  email          TEXT,
  preferred_time TEXT,
  notes          TEXT,
  source         TEXT DEFAULT 'chat',
  stage          TEXT,
  done           BOOLEAN DEFAULT FALSE,
  done_at        TIMESTAMPTZ,
  created_at     TIMESTAMPTZ DEFAULT now(),
  updated_at     TIMESTAMPTZ DEFAULT now()
);

-- New columns added safely if table already existed
ALTER TABLE corah_store.leads
  ADD COLUMN IF NOT EXISTS lead_ref TEXT;

-- Unique indexes for clean upserts / lookups
CREATE UNIQUE INDEX IF NOT EXISTS corah_leads_session_id_uq
  ON corah_store.leads(session_id);

CREATE UNIQUE INDEX IF NOT EXISTS corah_leads_lead_ref_uq
  ON corah_store.leads(lead_ref);
"""

UPSERT = """
INSERT INTO corah_store.leads (
  session_id, name, phone, email, preferred_time, notes, source, stage, done, done_at, lead_ref, updated_at
) VALUES (
  %(session_id)s, %(name)s, %(phone)s, %(email)s, %(preferred_time)s, %(notes)s, %(source)s, %(stage)s, %(done)s, %(done_at)s, %(lead_ref)s, now()
)
ON CONFLICT (session_id) DO UPDATE
SET
  name           = COALESCE(EXCLUDED.name,           corah_store.leads.name),
  phone          = COALESCE(EXCLUDED.phone,          corah_store.leads.phone),
  email          = COALESCE(EXCLUDED.email,          corah_store.leads.email),
  preferred_time = COALESCE(EXCLUDED.preferred_time, corah_store.leads.preferred_time),
  notes          = COALESCE(EXCLUDED.notes,          corah_store.leads.notes),
  source         = COALESCE(EXCLUDED.source,         corah_store.leads.source),
  stage          = COALESCE(EXCLUDED.stage,          corah_store.leads.stage),
  done           = COALESCE(EXCLUDED.done,           corah_store.leads.done),
  done_at        = COALESCE(EXCLUDED.done_at,        corah_store.leads.done_at),
  lead_ref       = COALESCE(EXCLUDED.lead_ref,       corah_store.leads.lead_ref),
  updated_at     = now()
RETURNING id;
"""

SELECT_ONE = """
SELECT
  id, session_id, name, phone, email, preferred_time, notes,
  source, stage, done, done_at, lead_ref, created_at, updated_at
FROM corah_store.leads
WHERE session_id = %(session_id)s
LIMIT 1;
"""


# ---------------------------
# Internals
# ---------------------------
def _ensure_schema() -> None:
    with pg_cursor(DB_URL) as cur:
        cur.execute(DDL)


def _new_lead_ref() -> str:
    return f"LEAD-{uuid.uuid4().hex[:8].upper()}"


def _sanitize_params(d: Dict[str, Any]) -> Dict[str, Any]:
    # convert empty strings to None so we never overwrite with ""
    for k, v in list(d.items()):
        if isinstance(v, str) and v.strip() == "":
            d[k] = None
    return d


# ---------------------------
# Public API
# ---------------------------
def save_lead(
    session_id: str,
    name: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    preferred_time: Optional[str] = None,
    notes: Optional[str] = None,
    source: Optional[str] = "chat",
    stage: Optional[str] = None,
    done: Optional[bool] = None,
    done_at: Optional[str] = None,   # ISO timestamp accepted
    lead_ref: Optional[str] = None,
) -> int:
    """
    Upsert one row per session_id.
    - Only non-null fields overwrite (COALESCE in UPSERT).
    - Idempotent: safe to call multiple times per stage.
    - Returns numeric lead id.
    """
    _ensure_schema()
    params = _sanitize_params({
        "session_id": session_id,
        "name": name,
        "phone": phone,
        "email": email,
        "preferred_time": preferred_time,
        "notes": notes,
        "source": source,
        "stage": stage,
        "done": done,
        "done_at": done_at,
        "lead_ref": lead_ref,
    })
    with pg_cursor(DB_URL) as cur:
        cur.execute(UPSERT, params)
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0


def get_lead(session_id: str) -> Optional[Dict[str, Any]]:
    _ensure_schema()
    with pg_cursor(DB_URL) as cur:
        cur.execute(SELECT_ONE, {"session_id": session_id})
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))


def mark_stage(session_id: str, stage: str, **kwargs: Any) -> int:
    """
    Persist stage transition + any provided fields (non-nulls only).
    Example: mark_stage(sid, "contact", phone=phone)
    """
    return save_lead(session_id, stage=stage, **kwargs)


def mark_done(session_id: str, **kwargs: Any) -> int:
    """
    Mark lead as done (one-shot).
    - Forces stage='done'
    - Sets done=True
    - Sets done_at (now) if not provided
    - Assigns a lead_ref if missing
    """
    lead = get_lead(session_id) or {}
    lead_ref = lead.get("lead_ref") or _new_lead_ref()
    done_at = kwargs.get("done_at")

    return save_lead(
        session_id,
        stage="done",
        done=True,
        done_at=done_at,   # generator/caller may pass an ISO; if None, UPSERT keeps existing or null
        lead_ref=lead_ref,
        **{k: v for k, v in kwargs.items() if k not in {"stage", "done", "lead_ref"}}
    )