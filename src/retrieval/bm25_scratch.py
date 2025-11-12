from __future__ import annotations
import math
from collections import Counter
from typing import List, Sequence

class BM25Okapi:
    """Minimal BM25 Okapi for understanding; not optimized."""
    def __init__(self, corpus_tokens: List[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = sum(self.doc_len) / max(1, self.N)
        self.tf = [Counter(d) for d in corpus_tokens]
        df = Counter()
        for c in self.tf:
            for t in c.keys():
                df[t] += 1
        # Robertson/Sparck Jones idf with +1 to keep positive
        self.idf = {t: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()}

    def get_scores(self, query_tokens: Sequence[str]) -> List[float]:
        scores = [0.0] * self.N
        for i, tf_d in enumerate(self.tf):
            denom = 1 - self.b + self.b * (self.doc_len[i] / max(1, self.avgdl))
            for t in query_tokens:
                if t not in tf_d:
                    continue
                idf = self.idf.get(t, 0.0)
                freq = tf_d[t]
                num = freq * (self.k1 + 1.0)
                den = freq + self.k1 * denom
                scores[i] += idf * (num / max(1e-9, den))
        return scores
