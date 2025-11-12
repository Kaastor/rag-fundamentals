from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field

class OutCitation(BaseModel):
    id: str
    title: Optional[str] = None
    anchor: Optional[str] = None

class OutSchema(BaseModel):
    answer: str = Field(...)
    citations: List[OutCitation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
