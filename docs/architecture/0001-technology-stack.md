# ADR 0001: Technology stack

- Status: Accepted
- Date: 2026-09-11

## Context

The system must answer SRE and Kubernetes questions that mix semantic intent with exact tokens
such as exit codes, CLI flags, error states, and YAML keys. It must remain testable without live
model services and expose retrieval quality, latency, token usage, and cost.

## Decision

- Use Python 3.11 with `uv`, FastAPI, Pydantic v2, Ruff, mypy, pytest, and pre-commit.
- Express the retrieval and generation workflow as explicit LangGraph nodes.
- Parse Markdown documents through a Docling adapter with source-specific normalization.
- Generate dense embeddings with `BAAI/bge-large-en-v1.5` through Sentence Transformers and
  PyTorch MPS on the development Mac.
- Combine Rank-BM25 sparse results and Qdrant dense results with a tested Python RRF
  implementation; later compare it with Qdrant-native RRF.
- Implement a shared reranker protocol with `cross-encoder/ms-marco-MiniLM-L6-v2` and
  `BAAI/bge-reranker-v2-m3` backends.
- Use OpenAI `gpt-5.6-luna` for hosted generation and `gpt-5.6-terra` for Ragas evaluation.
- Use native Ollama with `qwen3:8b` as the local generation fallback.
- Use OpenTelemetry instrumentation, Langfuse Cloud for LLM traces, and Prometheus/Grafana for
  service metrics.
- Gate changes with deterministic retrieval checks and Ragas evaluations in GitHub Actions.
- Run the API and PyTorch models natively during Apple Silicon development; run supporting
  infrastructure through ARM64-compatible Docker Compose services.

## Dataset scope

- Pin an upstream commit from `kubernetes/website` and ingest English `concepts/`, `tasks/`, and a
  filtered subset of `reference/`.
- Ingest the summaries and source links from `danluu/post-mortems`; do not crawl linked articles.
- Preserve repository, revision, source path, heading path, URL, license, and content hash in chunk
  metadata.
- Use stable chunk IDs in the golden benchmark and numbered inline citations mapped to structured
  citation records.

## Consequences

- Model, store, and telemetry providers require interfaces so unit tests can use local fakes.
- MPS acceleration is available only to native macOS processes in the development topology.
- Cloud telemetry must redact or truncate source content unless explicitly enabled.
- The CI gate includes context precision and context recall of at least `0.80`, faithfulness of at
  least `0.85`, and complete citation, schema, exact-token, and refusal validation.
