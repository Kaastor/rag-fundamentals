from __future__ import annotations
import os, json, pathlib
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = pathlib.Path("indexes")
FAISS_PATH = INDEX_DIR / "corpus.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"

class Retriever:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self._load()

    def _load(self):
        self.chunks: List[Dict] = []
        with CHUNKS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))
        self.texts = [c["text"] for c in self.chunks]
        self.ids = [c["id"] for c in self.chunks]
        self.by_id = {c["id"]: c for c in self.chunks}

        self.model = SentenceTransformer(self.embedding_model)

        # Optional from-scratch cosine index for learning
        self._use_simple = os.getenv("SIMPLE_INDEX", "0") == "1"
        if self._use_simple:
            from .simple_index import SimpleIndex
            embs = self.model.encode(self.texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            self.index = SimpleIndex(embs)
        else:
            self.index = faiss.read_index(str(FAISS_PATH))

        # Tokenizer for BM25
        tok = lambda s: [t for t in s.lower().split()]
        self._tok = tok

        # Optional from-scratch BM25 for learning
        self._bm25_impl = "rank_bm25"
        if os.getenv("BM25_SCRATCH", "0") == "1":
            from .bm25_scratch import BM25Okapi as _BM25
            self.bm25 = _BM25([tok(t) for t in self.texts])
            self._bm25_impl = "scratch"
        else:
            from rank_bm25 import BM25Okapi as _BM25
            self.bm25 = _BM25([tok(t) for t in self.texts])

        # Load index meta if present
        meta_path = INDEX_DIR / "meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def embed(self, q: str):
        v = self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
        return v.astype(np.float32)

    def topk_embeddings(self, q: str, k: int = 4) -> List[Dict]:
        v = self.embed(q)
        if self._use_simple:
            scores, idxs = self.index.search(v, k)
            scores = scores[:k]; idxs = idxs[:k]
        else:
            scores, idxs = self.index.search(v, k)
            scores = scores[0]; idxs = idxs[0]
        out = []
        for s, i in zip(scores, idxs):
            c = self.chunks[int(i)]
            out.append({**c, "emb": float(s), "bm25": None, "ranker": "emb"})
        return out

    def topk_bm25(self, q: str, k: int = 4) -> List[Dict]:
        scores = self.bm25.get_scores(self._tok(q))
        idxs = np.argsort(scores)[::-1][:k]
        out = []
        for i in idxs:
            c = self.chunks[int(i)]
            out.append({**c, "bm25": float(scores[int(i)]), "emb": None, "ranker": f"bm25:{self._bm25_impl}"})
        return out

    def hybrid_union(self, q: str, k: int = 4, tie_breaker: str = "bm25") -> List[Dict]:
        bm = self.topk_bm25(q, k)
        em = self.topk_embeddings(q, k)
        by: Dict[str, Dict] = {}
        for item in bm + em:
            d = by.get(item["id"], {**item, "bm25": item.get("bm25"), "emb": item.get("emb")})
            if item.get("bm25") is not None:
                d["bm25"] = item["bm25"]
            if item.get("emb") is not None:
                d["emb"] = item["emb"]
            by[item["id"]] = d
        items = list(by.values())
        key = "bm25" if tie_breaker == "bm25" else "emb"
        items.sort(key=lambda x: (float("-inf") if x.get(key) is None else x.get(key)), reverse=True)
        return items[:k]
