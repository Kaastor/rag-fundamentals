# Steel-Thread DocBot

Minimal, end-to-end RAG slice for labs: CLI baseline, FAISS + BM25 retrieval, τ-refusal via bigram support, eval harness, and safety mini-suite.

---

## Contents

- [Overview](#overview)
- [Repo layout](#repo-layout)
- [Requirements](#requirements)
- [Install](#install)
- [First run (index the corpus)](#first-run-index-the-corpus)
- [Ways to run it](#ways-to-run-it)
- [Configuration](#configuration)
- [Evaluation & safety](#evaluation--safety)
- [Troubleshooting](#troubleshooting)
- [Quick checklist](#quick-checklist)

---

## Overview

- **Retriever:** Sentence-Transformers embeddings + FAISS; BM25 union (hybrid) with deterministic tie‑breaking.
- **Generation:** Jinja-rendered prompt, JSON-only answers, and **support-score gate** (n‑gram overlap) with τ threshold → safe refusals when evidence is thin.
- **Entry points:** CLI (Typer) for indexing, RAG queries, and evaluation.
- **Eval:** Devset scoring, safety mini-suite, and auto-selection of τ that meets guardrails.

---

## Repo layout

```
.
├─ steel-thread/
│  ├─ ADR.md                        # Architecture decisions
│  ├─ README.md                     # (this file)
│  ├─ pyproject.toml                # Poetry project + console scripts
│  ├─ src/
│  │  ├─ cli.py                     # Typer CLI (index, rag, eval, ...)
│  │  ├─ pipeline.py                # Orchestration + τ gating
│  │  ├─ config.py                  # Settings (env-driven)
│  │  ├─ schemas.py                 # Output schema (answer/citations/confidence)
│  │  ├─ prompts/answer_prompt.jinja# JSON-only prompt template
│  │  ├─ clients/model_client.py    # Groq chat completion client
│  │  ├─ retrieval/                 # FAISS/BM25/Hybrid + indexer
│  │  ├─ metrics/                   # support_score, citation checks
│  │  └─ utils/text.py              # helpers (F1, normalize)
│  ├─ data/
│  │  ├─ corpus/*.md                # Example docs (index these)
│  │  ├─ devset.jsonl               # Q/A ground truth
│  │  └─ safety_prompts.jsonl       # Safety probes
│  ├─ tests/test_retriever.py       # Basic retrieval tests
│  ├─ indexes/                      # (generated) FAISS + chunks.jsonl + meta.json
│  └─ logs/                         # (generated) run.jsonl + experiments.csv
└─ diagrams/
   ├─ indexing.md
   ├─ rag.md
   └─ runtime.md
```

> **Note:** `indexes/` and `logs/` are created at runtime.

---

## Requirements

- **Python:** 3.11 (exact version required)
- **Poetry:** for dependency & script management
- CPU-only is fine (`faiss-cpu`). Sentence-Transformers will download the embedding model on first run.

---

## Install

```bash
# from repo root
cd steel-thread
poetry install
```

Console script is registered via `pyproject.toml` under `[tool.poetry.scripts]`:
- `docbot`

---

## First run (index the corpus)

You **must** build an index before asking questions.

```bash
poetry run docbot index --embedding-model all-MiniLM-L6-v2
# writes: indexes/corpus.faiss, indexes/chunks.jsonl, indexes/meta.json
```

Indexing splits `data/corpus/*.md` into blocks, embeds with Sentence-Transformers (normalized), and stores an inner‑product FAISS index.

---

## Ways to run it

### 1) CLI (development)

```bash
# Non-RAG (model only; should usually refuse)
poetry run docbot baseline -q "Where is the configuration file stored?"

# RAG (hybrid union by default; k=4; τ=0.4; tie-breaker=bm25)
poetry run docbot rag -q "Where is the configuration file stored?"

# Evaluate (devset + safety; logs experiments.csv)
poetry run docbot eval

# Safety sweep (prints per-case pass/fail)
poetry run docbot safety
```

`docbot rag` queries FAISS and BM25, unions + dedups results, prompts the model for **JSON** output, then **rejects** answers that don’t clear the support threshold τ or that lack citations.

---

## Configuration

All runtime config is environment-driven (see `src/config.py` and `src/clients/model_client.py`). A `.env` file is **not** auto-loaded.

### Core

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq API key for chat completions. |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model ID (e.g., `llama-3.3-70b-versatile`). |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers name for indexing & queries. |
| `K` | `4` | Top-k contexts. |
| `TAU` | `0.4` | Support gate threshold (0–1) on bigram coverage. |
| `MAX_TOKENS` | `512` | Generation cap. |
| `TEMPERATURE` | `0.2` | Decoding temperature. |

### Retrieval toggles

| Variable | Effect |
|---|---|
| `SIMPLE_INDEX=1` | Use a pure-NumPy cosine index (`SimpleIndex`) instead of FAISS. |
| `BM25_SCRATCH=1` | Use the from-scratch BM25 implementation instead of `rank_bm25`. |

---

## Evaluation & safety

- **Devset:** `data/devset.jsonl` (Q/A with expected supporting docs).
- **Safety suite:** `data/safety_prompts.jsonl` (prompts expecting either refusal or supported answer).
- **Auto-τ scan:** `poetry run docbot eval` tries τ in `(0.2, 0.4, 0.6)` and chooses the smallest τ that yields valid citations for all dev answers **and** ≥7/8 safety passes. Results are appended to `logs/experiments.csv`.

---

## Troubleshooting

- **“File not found” for FAISS/chunks:** Run `poetry run docbot index` first. Retrieval loads `indexes/chunks.jsonl` and `indexes/corpus.faiss`.
- **Model/API issues:** Ensure `GROQ_API_KEY` is set correctly. The client uses Groq's OpenAI-compatible API.
- **Over‑eager answers:** Increase `TAU` or ensure your corpus actually contains support. The gate enforces “answer only with evidence.”
- **Performance/cost:** Lower `K`, shorten questions, or reduce `MAX_TOKENS`.

---

## Quick checklist

1. `poetry install`
2. `poetry run docbot index`
3. Run queries:
   - `poetry run docbot rag -q "..."`
4. Optional: `poetry run docbot eval` → adjust `TAU`, `K`, model as needed.