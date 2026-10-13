# Product dashboard

The Streamlit dashboard is a thin client of the FastAPI service. It never imports retrieval models,
connects directly to Qdrant, or generates an answer itself. This keeps the user interface replaceable
and prevents UI reruns from loading expensive PyTorch models more than once.

## Capabilities

- Chat-style questions that preserve exact tokens such as `OOMKilled`, `137`, and `maxUnavailable`
- Grounded answers with clickable, structured citations
- Explicit guardrail refusals with stable machine-readable codes
- Request latency, generation provider, token count, and retained-evidence count
- Expandable source chunks with the rank and score assigned by each retrieval stage
- API health and a link to the separate Grafana operations dashboard

Grafana answers operational questions such as “Is latency degrading?” Streamlit answers product and
debugging questions such as “What did the system retrieve, cite, and return for this query?”

## Run locally

Install every dependency group and start the supporting services:

```bash
uv sync --all-groups
docker compose up -d
```

Start the FastAPI process:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run uvicorn sre_rag.main:create_runtime_app \
  --factory --host 0.0.0.0 --port 8000
```

For live queries, deployment composition must inject a real `RAGWorkflow` as described under
**Runtime readiness** below. Start the dashboard in another terminal:

```bash
uv run --group dashboard streamlit run src/sre_rag/dashboard/app.py
```

Open <http://127.0.0.1:8501>. The API address defaults to `http://localhost:8000` and can be changed
from the dashboard sidebar or with `SRE_RAG_API_BASE_URL`.

## Runtime readiness

The production factory loads the already-built, verified index during application lifespan. It never
indexes an unpinned corpus at startup. `/readyz` returns success only after the workflow is loaded;
startup fails closed if sparse and dense index state disagree.

The public query response includes only citation-validated answers or deterministic refusals, final
reranked evidence, aggregate latency, and safe model usage. Raw prompts, unvalidated model drafts,
provider response IDs, and secrets are not returned.
