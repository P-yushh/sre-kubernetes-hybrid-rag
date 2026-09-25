"""Headless smoke test for the optional Streamlit interface."""

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_dashboard_renders_offline_state_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SRE_RAG_API_BASE_URL", "http://127.0.0.1:9")
    app_path = Path(__file__).parents[2] / "src/sre_rag/dashboard/app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "SRE & Kubernetes Hybrid RAG"
    assert app.sidebar.error[0].value == "API offline"
    assert app.chat_input[0].placeholder.startswith("Ask about Kubernetes")
