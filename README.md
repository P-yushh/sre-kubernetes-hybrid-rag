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
docker compose up -d qdrant
uv run sre-rag-index
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run uvicorn sre_rag.main:create_runtime_app \
  --factory --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/readyz` for runtime readiness or
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

A source-grounded 30-case benchmark candidate is included for human review. It is deliberately not
promoted to the release filename, so the live gate cannot imply approval that has not happened.

Never commit `.env`, API keys, model caches, downloaded corpora, or generated vector data.

## Knowledge corpora

The Kubernetes and postmortem corpora remain remotely hosted and are pinned by full commit SHA in
[`corpora/manifest.yaml`](corpora/manifest.yaml). The verifier securely materializes only the selected
Markdown into temporary storage, validates its file count and content checksum, and deletes it when
finished:

```bash
uv run sre-rag-corpus-verify
```

See [`docs/corpus-management.md`](docs/corpus-management.md) for source attribution, licenses,
security limits, and the controlled source-update procedure.

The index command streams each verified corpus through ingestion, writes dense vectors to a
manifest-versioned Qdrant collection, then atomically publishes the checksummed sparse chunk
artifact. See [`docs/indexing-and-runtime.md`](docs/indexing-and-runtime.md) for the rebuild and
startup contracts.

## Architecture

The accepted technology choices and their rationale are recorded in
[`docs/architecture/0001-technology-stack.md`](docs/architecture/0001-technology-stack.md).
The golden-set schema, retrieval metrics, and faithfulness thresholds are documented in
[`docs/evaluation.md`](docs/evaluation.md).
The metrics and tracing model is documented in [`docs/observability.md`](docs/observability.md).
