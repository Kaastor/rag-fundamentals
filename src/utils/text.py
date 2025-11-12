from __future__ import annotations
import re

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def f1(a: str, b: str) -> float:
    A = normalize(a).split()
    B = normalize(b).split()
    if not A or not B:
        return 0.0
    inter = len(set(A) & set(B))
    prec = inter / max(1, len(A))
    rec  = inter / max(1, len(B))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)
