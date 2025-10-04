"""
app/api/schemas.py

Pydantic models for Corah's API layer.
Keep these concise and stable so clients don't break.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------
# Health
# ---------------------------

class HealthResponse(BaseModel):
    ok: bool = True


# ---------------------------
# Search (retrieval) models
# ---------------------------

class SearchHit(BaseModel):
    # Identifiers (may be None if not exposed by the query)
    document_id: Optional[int] = None
    chunk_no: Optional[int] = None

    # Metadata
    title: Optional[str] = None
    source_uri: Optional[str] = None  # e.g., s3 path or doc slug

    # Scoring
    score: float = Field(..., description="Blended relevance score (0..1)")

    # Content preview
    content: Optional[str] = None


class SearchResponse(BaseModel):
    hits: List[SearchHit] = []


# ---------------------------
# Chat (generation) models
# ---------------------------

class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description="Client session identifier for short-term memory."
    )
    question: str = Field(..., description="User's input text.")

    # Optional knobs (all are overrides; if None, server uses defaults)
    k: Optional[int] = Field(default=None, description="Top-k retrieval override.")
    max_context: Optional[int] = Field(default=None, description="Max chars for assembled context.")
    debug: Optional[bool] = Field(default=None, description="If true, include retrieval/trace in response.")
    citations: Optional[bool] = Field(default=None, description="If true, include simple citations list.")


class Citation(BaseModel):
    title: Optional[str] = None
    chunk_no: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    citations: Optional[List[Citation]] = None
    debug: Optional[Dict[str, Any]] = None