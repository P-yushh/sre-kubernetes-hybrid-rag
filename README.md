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
- Docker Desktop for Qdrant and local monitoring

## Local setup

```bash
cp .env.example .env
uv sync --dev
uv run uvicorn sre_rag.main:app --reload --host 0.0.0.0
```

Open `http://127.0.0.1:8000/healthz` for the health response or
`http://127.0.0.1:8000/docs` for the generated API documentation.

Qdrant, Prometheus, and Grafana run as ARM64-aware supporting containers while the API and PyTorch
models remain native for Apple MPS acceleration. See
[`docs/local-development.md`](docs/local-development.md) for startup commands, service URLs,
persistent storage behavior, and the provisioned operations dashboard.

## Product dashboard

The optional Streamlit interface displays grounded answers, citations, guardrail refusals, retrieval
scores, latency, provider, and token usage without coupling UI reruns to model execution:

```bash
uv sync --all-groups
uv run --group dashboard streamlit run src/sre_rag/dashboard/app.py
```

See [`docs/dashboard.md`](docs/dashboard.md) for the API boundary, local URL, and runtime-readiness
behavior. Streamlit provides query-level product inspection; Grafana remains the operations dashboard.

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
The metrics and tracing model is documented in [`docs/observability.md`](docs/observability.md).
