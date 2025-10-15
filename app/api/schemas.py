# app/api/schemas.py
"""
API-facing Pydantic models for Corah.

Includes:
- HealthResponse
- SearchHit / SearchResponse
- ChatRequest / ChatResponse

Design notes
- Keep models minimal and stable for the API layer.
- Allow flexible metadata on ChatResponse via a generic dict.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    ok: bool = Field(True, description="Service health flag")


class SearchHit(BaseModel):
    score: float = Field(..., description="Relevance score")
    content: str = Field(..., description="Snippet/content for the hit")
    source: Optional[str] = Field(None, description="Source identifier (file/url/etc.)")


class SearchResponse(BaseModel):
    hits: List[SearchHit] = Field(default_factory=list)


class ChatRequest(BaseModel):
    question: str = Field(..., description="User message")
    session_id: Optional[str] = Field(
        None,
        description="Session identifier; server will create one if not provided",
    )


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Assistant reply text")
    session_id: str = Field(..., description="Current session id")
    session_closed: bool = Field(
        False, description="True if the session is finished and input should be disabled"
    )
    meta: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Auxiliary metadata (sentiment, intent_level, priority, recap flags, etc.)",
    )