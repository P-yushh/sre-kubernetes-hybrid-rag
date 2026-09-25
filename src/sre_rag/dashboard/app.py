"""Interactive Streamlit interface for grounded SRE and Kubernetes search."""

from typing import Any

import streamlit as st

from sre_rag.api.query import EvidenceView, QueryResult
from sre_rag.config import Settings
from sre_rag.dashboard.client import DashboardAPIError, DashboardClient
from sre_rag.domain.answers import GroundedAnswer, RefusalResponse

EXAMPLE_QUESTIONS = (
    "Why was my Kubernetes container OOMKilled with exit code 137?",
    "How does maxUnavailable affect a rolling update?",
    "What causes CrashLoopBackOff and how should I investigate it?",
)


def main() -> None:
    """Render the standalone product interface."""

    st.set_page_config(
        page_title="SRE Hybrid RAG",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    settings = Settings()

    st.title("SRE & Kubernetes Hybrid RAG")
    st.caption(
        "Exact-token BM25 + semantic retrieval, reciprocal-rank fusion, neural reranking, "
        "and citation-validated answers."
    )

    api_url = _render_sidebar(settings.api_base_url)
    _render_history()

    prompt = st.chat_input(
        "Ask about Kubernetes errors, CLI flags, YAML fields, runbooks, or incidents"
    )
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Running hybrid retrieval and validating citations..."):
                try:
                    with DashboardClient(api_url) as client:
                        result = client.query(prompt)
                except DashboardAPIError as error:
                    st.error(str(error))
                else:
                    _store_turn(prompt, result)
                    _render_result(result)


def _render_sidebar(default_api_url: str) -> str:
    with st.sidebar:
        st.header("System")
        api_url = str(st.text_input("API URL", value=default_api_url))
        try:
            with DashboardClient(api_url, timeout_seconds=2.0) as client:
                health = client.health()
        except (DashboardAPIError, ValueError):
            st.error("API offline")
        else:
            st.success("API online")
            st.caption(f"{health.service} · {health.environment} · v{health.version}")

        st.divider()
        st.subheader("Try a question")
        for example in EXAMPLE_QUESTIONS:
            st.caption(f"• {example}")

        st.divider()
        st.link_button("Operational metrics", "http://127.0.0.1:3000")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.pop("query_history", None)
            st.rerun()
    return api_url


def _store_turn(question: str, result: QueryResult) -> None:
    history: list[dict[str, Any]] = st.session_state.setdefault("query_history", [])
    history.append({"question": question, "result": result.model_dump(mode="json")})


def _render_history() -> None:
    history = st.session_state.get("query_history", [])
    for turn in history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            _render_result(QueryResult.model_validate(turn["result"]))


def _render_result(result: QueryResult) -> None:
    response = result.response
    if isinstance(response, GroundedAnswer):
        st.markdown(response.answer)
        with st.container(border=True):
            st.caption("Citations")
            for citation in response.citations:
                st.markdown(
                    f"**[{citation.number}] {citation.source_title}**  \n"
                    f"[{citation.source_path}]({citation.source_url})"
                )
    elif isinstance(response, RefusalResponse):
        st.warning(response.message, icon="🛡️")
        st.caption(f"Guardrail code: `{response.code.value}`")

    generation = result.generation
    metric_columns = st.columns(4)
    metric_columns[0].metric("Latency", f"{result.latency_ms:,.0f} ms")
    metric_columns[1].metric("Evidence", len(result.contexts))
    metric_columns[2].metric("Provider", generation.provider.value if generation else "None")
    total_tokens = generation.input_tokens + generation.output_tokens if generation else 0
    metric_columns[3].metric("Tokens", total_tokens)

    with st.expander("Inspect retrieval evidence and ranking", expanded=False):
        if not result.contexts:
            st.info("No evidence survived retrieval and reranking.")
        for index, context in enumerate(result.contexts, start=1):
            _render_context(index, context)

    st.caption(f"Query ID: `{result.query_id}`")


def _render_context(index: int, context: EvidenceView) -> None:
    st.markdown(f"#### {index}. {context.title}")
    st.caption(f"Chunk `{context.chunk_id}` · [{context.source_path}]({context.source_url})")
    rows = [
        {"stage": stage.method.value, "rank": stage.rank, "score": stage.score}
        for stage in context.stages
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.text(context.text)
    st.divider()


if __name__ == "__main__":
    main()
