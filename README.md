# SRE Kubernetes Hybrid RAG

A production-oriented Retrieval-Augmented Generation service for Kubernetes documentation and
SRE incident knowledge. It combines exact-token BM25 retrieval with dense semantic search,
Reciprocal Rank Fusion, cross-encoder reranking, grounded answers, and measurable quality gates.

## Foundation

The service uses a typed FastAPI application factory, environment-based configuration, automated
quality checks, and a health endpoint. Retrieval and generation components are isolated behind
interfaces so they can be tested independently.

## Prerequisites

- macOS or Linux
- Python 3.11
- [`uv`](https://docs.astral.sh/uv/)

## Local setup

```bash
cp .env.example .env
uv sync --dev
uv run uvicorn sre_rag.main:app --reload
```

Open `http://127.0.0.1:8000/healthz` for the health response or
`http://127.0.0.1:8000/docs` for the generated API documentation.

## Quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest --cov=sre_rag --cov-report=term-missing
```

GitHub Actions runs the same deterministic checks for pull requests. The separate live RAG gate is
activated only when a reviewed golden dataset, runtime factory, and protected evaluation secret are
configured; see [`docs/evaluation.md`](docs/evaluation.md).

Never commit `.env`, API keys, model caches, downloaded corpora, or generated vector data.

## Architecture

The accepted technology choices and their rationale are recorded in
[`docs/architecture/0001-technology-stack.md`](docs/architecture/0001-technology-stack.md).
The golden-set schema, retrieval metrics, and faithfulness thresholds are documented in
[`docs/evaluation.md`](docs/evaluation.md).
