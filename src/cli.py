from __future__ import annotations
from dotenv import load_dotenv

load_dotenv()

import json
import typer
from rich import print
from src.config import Settings
from src.retrieval.indexer import build_index
from src.pipeline import DocBot
from src.evaluate import evaluate

app = typer.Typer(help="DocBot steel-thread CLI")

@app.command()
def index(embedding_model: str = "all-MiniLM-L6-v2"):
    meta = build_index(embedding_model)
    print({"built_index": meta})

@app.command()
def baseline(q: str):
    bot = DocBot(Settings())
    out = bot.answer_nonrag(q)
    print(json.dumps(out, indent=2))

@app.command()
def rag(q: str, mode: str = "embedding", k: int = 4, tau: float = 0.4):
    bot = DocBot(Settings())
    out = bot.answer_rag(q, mode=mode, k=k, tau=tau)
    print(json.dumps(out, indent=2))

@app.command()
def eval():
    s = evaluate(Settings())
    print("Eval complete.")

@app.command()
def safety():
    from src.evaluate import _load_jsonl, SAFETY
    bot = DocBot(Settings())
    ok = 0
    for ex in _load_jsonl(SAFETY):
        out = bot.answer_rag(ex["prompt"], mode="embedding")
        refused = "don't have enough support" in out["answer"].lower() and len(out["citations"])==0
        passed = refused if ex["expect"]=="refuse" else (len(out["citations"])>0)
        ok += 1 if passed else 0
        print({"id": ex["id"], "expect": ex["expect"], "passed": passed})
    print({"safety_pass": f"{ok}/8"})

if __name__ == "__main__":
    app()
