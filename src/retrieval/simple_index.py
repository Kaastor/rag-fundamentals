from __future__ import annotations
import numpy as np
from typing import Tuple

class SimpleIndex:
    """Tiny cosine-similarity index using only NumPy. For teaching."""
    def __init__(self, embs: np.ndarray):
        e = embs.astype(np.float32)
        norms = np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
        self.embs = e / norms  # (n, d) normalized

    def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = q.astype(np.float32)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        scores = self.embs @ q.T  # (n,1)
        scores = scores.ravel()
        if k >= len(scores):
            idxs = np.argsort(scores)[::-1]
        else:
            topk = np.argpartition(scores, -k)[-k:]
            idxs = topk[np.argsort(scores[topk])[::-1]]
        return scores[idxs], idxs
