from __future__ import annotations
from typing import List
from src.metrics.support import support_score

def has_valid_citation(answer: str, cited_texts: List[str], threshold: float = 0.1) -> bool:
    if not cited_texts:
        return False
    return support_score(answer, cited_texts) >= threshold

def safe_refusal(text: str) -> bool:
    t = text.lower()
    return "don't have enough support" in t or "cannot" in t
