from __future__ import annotations
import json, hashlib, pathlib, re
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = pathlib.Path("data/corpus")
INDEX_DIR = pathlib.Path("indexes")
META_PATH = INDEX_DIR / "meta.json"
FAISS_PATH = INDEX_DIR / "corpus.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"

@dataclass
class Chunk:
    id: str
    title: str
    source: str
    text: str

def _read_docs() -> List[tuple[str, str]]:
    docs = []
    for p in sorted(DATA_DIR.glob("*.md")):
        docs.append((p.name, p.read_text(encoding="utf-8")))
    return docs

def _split(md_text: str) -> List[str]:
    blocks = re.split(r"\n\s*\n", md_text.strip())
    return [b.strip() for b in blocks if b.strip()]

def build_index(embedding_model: str = "all-MiniLM-L6-v2") -> Dict:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    docs = _read_docs()
    chunks: List[Chunk] = []
    for fname, text in docs:
        title = fname.replace(".md", "")
        for i, block in enumerate(_split(text)):
            cid = f"{fname}::p{i:03d}"
            chunks.append(Chunk(id=cid, title=title, source=fname, text=block))

    model = SentenceTransformer(embedding_model)
    embs = model.encode([c.text for c in chunks], normalize_embeddings=True, convert_to_numpy=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))

    faiss.write_index(index, str(FAISS_PATH))
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__) + "\n")

    checksum = hashlib.sha256(("".join(s for _, s in docs)).encode()).hexdigest()[:10]
    meta = {"embedding_model": embedding_model, "dim": dim, "count": len(chunks), "corpus_sha": checksum}
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
