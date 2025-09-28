from __future__ import annotations
from pydantic import BaseModel
from typing import List, Optional, Any

class SearchHit(BaseModel):
    doc_id: int
    title: Optional[str] = None
    uri: Optional[str] = None
    chunk_id: int
    chunk_no: int
    content: str
    distance: float
    similarity: float

class SearchResponse(BaseModel):
    query: str
    k: int
    hits: List[SearchHit]

class ChatRequest(BaseModel):
    question: str
    k: int = 5
    max_context: int = 3000
    history: Optional[List[Any]] = None  # reserved for future memory

class ChatResponse(BaseModel):
    answer: str
    rewritten_query: Optional[str] = None
    used: List[SearchHit]