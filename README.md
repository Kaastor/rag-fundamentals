# DocBot

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

- **Retriever:** Sentence-Transformers embeddings + FAISS; BM25 or Dense retrieval (selectable mode).
- **Generation:** Jinja-rendered prompt, JSON-only answers, and **support-score gate** (n‑gram overlap) with τ threshold → safe refusals when evidence is thin.
- **Entry points:** CLI (Typer) for indexing, RAG queries, and evaluation.
- **Eval:** Devset scoring, safety mini-suite, and auto-selection of τ that meets guardrails.

---

## Repo layout

```
.
├─ steel-thread/
│  ├─ README.md                     # (this file)
│  ├─ pyproject.toml                # Poetry project + console scripts
│  ├─ decisions_log.md              # Architecture decisions log
│  ├─ .env.example                  # Example environment variables
│  ├─ src/
│  │  ├─ cli.py                     # Typer CLI (index, rag, eval, ...)
│  │  ├─ pipeline.py                # Orchestration + τ gating
│  │  ├─ config.py                  # Settings (env-driven)
│  │  ├─ schemas.py                 # Output schema (answer/citations/confidence)
│  │  ├─ evaluate.py                # Evaluation & safety harness
│  │  ├─ prompts/answer_prompt.jinja# JSON-only prompt template
│  │  ├─ clients/model_client.py    # Groq chat completion client
│  │  ├─ retrieval/                 # FAISS/BM25 + indexer
│  │  │  ├─ indexer.py              # Index builder
│  │  │  ├─ retriever.py            # Retrieval logic
│  │  │  └─ simple_index.py         # Pure NumPy index (optional)
│  │  ├─ metrics/                   # support_score, citation checks
│  │  └─ utils/text.py              # helpers (F1, normalize)
│  ├─ data/
│  │  ├─ corpus/*.md                # Example docs (index these)
│  │  ├─ devset.jsonl               # Q/A ground truth
│  │  └─ safety_prompts.jsonl       # Safety probes
│  └─ indexes/                      # (generated) FAISS + chunks.jsonl + meta.json
```

> **Note:** `indexes/` directory is created at runtime during first indexing.

---

## Requirements

- **Python:** 3.11.13 (exact version specified in `pyproject.toml`)
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
# output: {'built_index': {'embedding_model': 'all-MiniLM-L6-v2', 'dim': 384, 'count': 19, 'corpus_sha': '59258e0a37'}}
```

Indexing splits `data/corpus/*.md` into blocks, embeds with Sentence-Transformers (normalized), and stores an inner‑product FAISS index.

---

## Ways to run it

### 1) CLI (development)

```bash
# Non-RAG (model only; should usually refuse)
poetry run docbot baseline "Where is the configuration file stored?"
poetry run docbot baseline "What is the default for 'level'?"

# RAG (embedding mode by default; k=4; τ=0.4)
poetry run docbot rag "Where is the configuration file stored?"
poetry run docbot rag "What is the default for 'level'?" --mode bm25


# Evaluate (devset + safety; logs experiments.csv)
poetry run docbot eval

# Safety sweep (prints per-case pass/fail)
poetry run docbot safety
```


**`docbot rag` options:**
- `-q, --q TEXT` - Query string (required)
- `--mode TEXT` - Retrieval mode: "embedding" or "bm25" (default: "embedding")
- `--k INTEGER` - Number of top contexts to retrieve (default: 4)
- `--tau FLOAT` - Support score threshold for refusal gating (default: 0.4)

`docbot rag` queries FAISS or BM25 based on mode, prompts the model for **JSON** output, then **rejects** answers that don’t clear the support threshold τ or that lack citations.

---

## Configuration

All runtime config is environment-driven (see `src/config.py` and `src/clients/model_client.py`). 

You can set environment variables directly or create a `.env` file in the project root (see `.env.example` for reference).

### LLM API Key

- Create an account: `https://console.groq.com/home`
- Create API key: `https://console.groq.com/keys` (recomendation: store in password manager)
- Add it in `.env` file

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
   - `poetry run docbot rag "..."`
4. Optional: `poetry run docbot eval` → adjust `TAU`, `K`, model as needed.