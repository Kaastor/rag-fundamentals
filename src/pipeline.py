from __future__ import annotations
import json
from typing import Dict, Any, List
from jinja2 import Environment, FileSystemLoader, select_autoescape
from src.config import Settings
from src.clients.model_client import ModelClient, ClientConfig
from src.retrieval.retriever import Retriever
from src.metrics.support import support_score

env = Environment(loader=FileSystemLoader("src/prompts"), autoescape=select_autoescape())

def _render_prompt(question: str, contexts: List[Dict]) -> str:
    tpl = env.get_template("answer_prompt.jinja")
    return tpl.render(question=question, contexts=contexts)

class DocBot:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.retriever = Retriever(cfg.embedding_model)
        self.client = ModelClient(ClientConfig(model_id=cfg.model_id))

    def answer_nonrag(self, question: str) -> Dict[str, Any]:
        prompt = _render_prompt(question, [])
        res = self.client.generate_json(prompt)
        try:
            payload = json.loads(res["text"])
        except Exception:
            payload = {"answer":"I don't have enough support to answer.","citations":[],"confidence":0.0}
        return payload

    def answer_rag(self, question: str, mode: str = "embedding", tau: float | None = None, k: int | None = None) -> Dict[str, Any]:
        k = k if k is not None else self.cfg.k
        tau = tau if tau is not None else self.cfg.tau

        if mode == "bm25":
            contexts = self.retriever.topk_bm25(question, k)
        else:
            contexts = self.retriever.topk_embeddings(question, k)
            
        prompt = _render_prompt(question, contexts)
        res = self.client.generate_json(prompt)
        try:
            payload = json.loads(res["text"])
        except Exception:
            payload = {"answer":"I don't have enough support to answer.","citations":[],"confidence":0.0}

        cited_texts = []
        by_id = self.retriever.by_id
        for c in (payload.get("citations") or []):
            if isinstance(c, dict) and c.get("id") in by_id:
                cited_texts.append(by_id[c["id"]]["text"])
        s = support_score(payload.get("answer",""), cited_texts if cited_texts else [c["text"] for c in contexts])
        if s < tau or not payload.get("citations"):
            payload = {"answer":"I don't have enough support to answer.","citations":[], "confidence": 0.05}

        return payload

def answer(msg: str) -> Dict[str, Any]:
    bot = DocBot(Settings())
    return bot.answer_rag(msg, mode="embedding")
