# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GGF LLM Systems Case v2.0 — a technical assessment for LLM systems engineering. The system uses an LLM (OpenAI-compatible) to generate unified diff patches for a TypeScript mini-game codebase based on natural language task descriptions. Scored on a 100-point rubric across 4 phases: RAG (30pts), Prompt Engineering (20pts), Fine-Tuning (30pts), Analytics (20pts).

## Repository Layout

Three distinct sub-projects, each with its own language/runtime:

- **`ggf-mini-game/`** — TypeScript target codebase (~3.5KB). Pure functional state management with reducers. This is what gets patched.
- **`solution/`** — Python solution framework (`src/ggf_case/`). Contains the RAG pipeline, LLM client, patch generation, fine-tuning, and analytics logic.
- **`eval/`** — Node.js evaluation harness. `tasks.json` defines 10 tasks; phase check scripts validate results. Gold labels in `gold_labels.json`.

## Commands

### TypeScript mini-game
```bash
cd ggf-mini-game
npm install
npm run build        # tsc compile
npm run typecheck    # tsc --noEmit
npm run demo         # node dist/demo.js (5-tick simulation)
```

### Python solution
```bash
cd solution
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # all extras (embeddings, analytics, dev)
cp ../.env.example ../.env       # then edit with real keys
```

CLI entry point is `ggf-case` (registered via pyproject.toml `[project.scripts]`):
```bash
ggf-case check-health           # verify LLM endpoint
ggf-case index                  # index mini-game codebase into index.json
ggf-case metrics                # retrieval metrics against gold labels
ggf-case run-eval               # run all 10 tasks
ggf-case run-task task_01       # run a single task
ggf-case finetune prepare       # curate training data
ggf-case finetune run <file>    # start OpenAI fine-tuning job
ggf-case finetune eval          # check job status
ggf-case analyze <results.json> # failure analysis
ggf-case report <results_dir>   # full evaluation report
```

### Evaluation checks
```bash
node eval/checks/baseline_sanity.mjs
node eval/phase_checks/phase1_rag.mjs
node eval/phase_checks/phase2_prompting.mjs
node eval/phase_checks/phase3_finetune.mjs
node eval/phase_checks/phase4_analytics.mjs
```

### Testing & linting (Python)
```bash
pytest                           # from solution/
ruff check src/                  # lint
ruff format src/                 # format
```

## Architecture

### Data Flow
1. **Indexing** (`rag/indexer.py`) — reads TypeScript source, produces `CodeChunk` objects via fixed-window or AST-aware chunking
2. **Retrieval** (`rag/retriever.py`, `rag/bm25.py`, `rag/hybrid.py`) — keyword, BM25, embedding, or hybrid (RRF) strategies; optional cross-encoder reranking (`rag/reranker.py`)
3. **Prompt Construction** (`llm/prompts.py`) — builds system + user messages with retrieved context
4. **LLM Call** (`llm/openai_compat.py`) — OpenAI-compatible API via httpx; structured output via Pydantic models (`llm/structured_output.py`)
5. **Patch Validation** (`patch/diff_guard.py`) — enforces max 250 lines, 6 files
6. **Patch Application** (`patch/apply_patch.py`) — `git apply` with fallback
7. **Evaluation** (`eval/runner.py`) — orchestrates tasks, compiles TypeScript, runs Node.js checks
8. **Metrics** (`metrics/retrieval_metrics.py`, `metrics/patch_metrics.py`) — precision@k, MRR, NDCG, exact/hunk match

### Fine-Tuning Pipeline
- `finetune/data_curator.py` — loads `eval/training_data/examples.jsonl` (50 examples), validates, splits train/val, exports OpenAI format
- `finetune/trainer.py` — uploads file, creates fine-tuning job via OpenAI API
- `finetune/evaluator.py` — compares base vs fine-tuned model

### Analytics
- `analytics/failure_analyzer.py` — classifies failures (retrieval_miss, generation_error, patch_apply_error, etc.)
- `analytics/experiment.py` — A/B testing with statistical significance

## Configuration

All settings via `.env` (see `.env.example`). Key variables:
- `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` — LLM endpoint config
- `TOP_K`, `EMBEDDING_MODEL` — RAG retrieval settings
- `DIFF_MAX_LINES=250`, `DIFF_MAX_FILES=6` — patch size limits
- `QDRANT_URL`, `QDRANT_COLLECTION` — optional vector DB (docker-compose.yml provides Qdrant)

Settings are loaded via `pydantic-settings` in `config.py`. The CLI auto-detects repo root by looking for `eval/tasks.json`.

## Key Constraints

- Python >= 3.11 required
- Node.js >= 18 required
- TypeScript compiled with strict mode, ES2020 target
- Patches must be unified diff format and pass `diff_guard` limits
- The 10 tasks range from easy (pause toggle) to hard (deterministic RNG, event log system)
