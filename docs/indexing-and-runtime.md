# Indexing and runtime contract

Corpus acquisition and indexing are explicit operator actions. The API never downloads source
repositories, parses Markdown, or mutates Qdrant while handling startup or a user query.

## Build the hybrid index

Start Qdrant, then run the offline build on the native macOS Python environment so PyTorch can use
Apple MPS:

```bash
docker compose up -d qdrant
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run sre-rag-index
```

The command performs these operations in order:

1. Download each full-SHA-pinned archive into a temporary directory.
2. Reject unsafe archive members and verify selected file counts and content checksums.
3. Normalize and heading-aware chunk the Markdown with stable chunk IDs.
4. Embed and upsert every chunk into a Qdrant collection whose suffix identifies the corpus
   manifest.
5. Atomically publish `data/processed/chunks.jsonl` and its checksummed
   `index-manifest.json` only after dense indexing succeeds.
6. Delete the temporary raw corpora.

Generated artifacts and downloaded source text are local runtime data and must not be committed.

## Start the query service

Set `SRE_RAG_OPENAI_API_KEY` in `.env`, ensure Ollama is serving the configured fallback model, and
start the lifespan-managed application:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run uvicorn sre_rag.main:create_runtime_app \
  --factory --host 0.0.0.0 --port 8000
```

Startup loads BM25 from the checksummed chunk artifact, loads the pinned embedding and reranker
models once, and connects dense retrieval to the collection named by the index manifest. Startup
fails if the local artifact is missing or corrupt, the collection is missing, its vector dimension
differs, or its point count does not exactly match the chunk count.

`/healthz` proves only that the HTTP process is alive. `/readyz` returns success only after the full
query workflow has loaded. Kubernetes or another orchestrator should use readiness—not liveness—to
decide whether to send traffic.

## Why the collection is versioned

An in-place rebuild can leave sparse and dense retrieval on different corpus revisions if the
process fails halfway through. A manifest-versioned collection isolates the new dense build. The
sparse artifact is published last and names that exact collection, so the runtime either loads a
consistent pair or refuses to start. Old collections may be removed later through an explicit
retention procedure after the new index has been validated.
