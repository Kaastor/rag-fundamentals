from __future__ import annotations
import os, time
from typing import Dict, Any
from dataclasses import dataclass
import tiktoken
from openai import OpenAI

@dataclass
class ClientConfig:
    model_id: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))

class ModelClient:
    def __init__(self, cfg: ClientConfig):
        self.cfg = cfg
        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        self._client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )

    def _count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc:
            try:
                return len(self._enc.encode(text))
            except Exception:
                pass
        return max(1, len(text.split()))

    def generate_json(self, prompt: str, stop: list[str] | None = None) -> Dict[str, Any]:
        start = time.time()
        tokens_in = self._count(prompt)

        resp = self._client.chat.completions.create(
            model=self.cfg.model_id,
            temperature=self.cfg.temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.cfg.max_tokens,
            stop=stop,
        )
        out = resp.choices[0].message.content
        tokens_out = int(getattr(resp.usage, "completion_tokens", 0) or 0)
        latency_ms = int((time.time() - start) * 1000)
        return {
            "text": out,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
        }
