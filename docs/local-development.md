# Local development

The application and model workloads run natively on macOS so PyTorch can use the Apple M3 GPU
through MPS. Docker Compose runs only the stateful supporting services: Qdrant, Prometheus, and
Grafana. Ollama also remains native, while Langfuse Cloud receives privacy-safe traces when its
credentials are configured.

```text
macOS host                              Docker Compose
┌──────────────────────────────┐        ┌─────────────────────────────┐
│ FastAPI + LangGraph :8000    │───────▶│ Qdrant :6333/:6334          │
│ PyTorch models on MPS        │        │ Prometheus :9090            │
│ Ollama :11434 (fallback)     │◀───────│ Grafana :3000               │
└──────────────────────────────┘        └─────────────────────────────┘
             │
             └── privacy-safe traces ──▶ Langfuse Cloud
```

## Prerequisites

- Python 3.11 and `uv`
- Docker Desktop with Apple Silicon support
- Ollama if local generation fallback is required

## Start the local stack

Create the local environment file and set a non-empty `GRAFANA_ADMIN_PASSWORD`:

```bash
cp .env.example .env
uv sync --dev
```

Start the supporting services:

```bash
docker compose up -d
docker compose ps
```

Run the API natively in a second terminal. Binding to all host interfaces allows Prometheus in its
container to reach the `/metrics` endpoint through `host.docker.internal`.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run uvicorn sre_rag.main:app \
  --reload --host 0.0.0.0 --port 8000
```

| Component | Local URL |
| --- | --- |
| API documentation | <http://127.0.0.1:8000/docs> |
| API health | <http://127.0.0.1:8000/healthz> |
| API metrics | <http://127.0.0.1:8000/metrics> |
| Streamlit product dashboard | <http://127.0.0.1:8501> |
| Qdrant dashboard | <http://127.0.0.1:6333/dashboard> |
| Prometheus | <http://127.0.0.1:9090> |
| Grafana | <http://127.0.0.1:3000> |

Grafana provisions the Prometheus data source and the **SRE Kubernetes Hybrid RAG** operations
dashboard automatically. The dashboard shows service health, workflow P50/P95 latency, per-stage
P95 latency, answer and refusal outcomes, and generation token throughput.

Run the separate query-level product dashboard with:

```bash
uv run --group dashboard streamlit run src/sre_rag/dashboard/app.py
```

The Streamlit interface and runtime-readiness contract are documented in
[`dashboard.md`](dashboard.md).

## Operate and stop the stack

Inspect service state and logs with:

```bash
docker compose ps
docker compose logs --tail=100
```

Stop containers without removing indexed vectors or dashboard history:

```bash
docker compose down
```

The named volumes persist Qdrant, Prometheus, and Grafana data across restarts. Running
`docker compose down -v` also deletes those local volumes and their data, so use it only when a
complete reset is intentional.

## Local security boundary

All published container ports bind to `127.0.0.1`, Grafana requires an explicit password, and
anonymous dashboard access is disabled. Qdrant has no local API key in this development topology,
so do not expose these services outside the laptop. This Compose stack is a reproducible development
environment, not a production deployment manifest.
