from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass
class Settings:
    model_id: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    k: int = int(os.getenv("K", "4"))
    tau: float = float(os.getenv("TAU", "0.4"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
