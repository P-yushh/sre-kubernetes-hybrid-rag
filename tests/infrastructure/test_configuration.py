"""Contract tests for the reproducible local infrastructure stack."""

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parents[2]


def _yaml(path: str) -> dict[str, Any]:
    with (ROOT / path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert isinstance(payload, dict)
    return payload


def test_compose_runs_only_arm64_supporting_services_on_loopback() -> None:
    compose = _yaml("compose.yaml")
    services = compose["services"]

    assert set(services) == {"qdrant", "prometheus", "grafana"}
    assert services["qdrant"]["image"] == "qdrant/qdrant:v1.19.1"
    assert services["prometheus"]["image"] == "prom/prometheus:v3.13.3-distroless"
    assert services["grafana"]["image"] == "grafana/grafana:13.2.2"
    assert all(service["platform"] == "linux/arm64" for service in services.values())

    published_ports = [port for service in services.values() for port in service["ports"]]
    assert all(port.startswith("127.0.0.1:") for port in published_ports)
    assert (
        "${GRAFANA_ADMIN_PASSWORD:?"
        in services["grafana"]["environment"]["GF_SECURITY_ADMIN_PASSWORD"]
    )


def test_prometheus_scrapes_native_api_and_qdrant() -> None:
    prometheus = _yaml("infra/prometheus/prometheus.yml")
    jobs = {job["job_name"]: job for job in prometheus["scrape_configs"]}

    assert jobs["sre-rag-api"]["metrics_path"] == "/metrics"
    assert jobs["sre-rag-api"]["static_configs"][0]["targets"] == ["host.docker.internal:8000"]
    assert jobs["qdrant"]["static_configs"][0]["targets"] == ["qdrant:6333"]


def test_grafana_provisions_prometheus_and_operations_dashboard() -> None:
    datasource = _yaml("infra/grafana/provisioning/datasources/prometheus.yml")
    assert datasource["datasources"][0]["url"] == "http://prometheus:9090"
    assert datasource["datasources"][0]["uid"] == "prometheus"

    dashboard = json.loads(
        (ROOT / "infra/grafana/dashboards/sre-rag-operations.json").read_text(encoding="utf-8")
    )
    assert dashboard["uid"] == "sre-rag-operations"
    assert dashboard["editable"] is False

    titles = {panel["title"] for panel in dashboard["panels"]}
    assert {
        "Service health",
        "Workflow P50 latency",
        "Workflow P95 latency",
        "Answer rate",
        "Pipeline stage P95 latency",
        "Query outcomes",
        "Generation token throughput",
    } <= titles

    expressions = "\n".join(
        target["expr"] for panel in dashboard["panels"] for target in panel["targets"]
    )
    assert "sre_rag_stage_duration_seconds_bucket" in expressions
    assert "sre_rag_queries_total" in expressions
    assert "sre_rag_generation_tokens_total" in expressions
