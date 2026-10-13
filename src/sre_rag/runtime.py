"""Fail-closed composition of the production query runtime."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from prometheus_client import CollectorRegistry
from qdrant_client import QdrantClient

from sre_rag.config import Settings
from sre_rag.generation.fallback import FallbackGenerationBackend
from sre_rag.generation.ollama import OllamaConfig, OllamaGenerationBackend
from sre_rag.generation.openai import OpenAIConfig, OpenAIGenerationBackend
from sre_rag.generation.prompting import GroundedDraftGenerator, GroundedGenerationConfig
from sre_rag.guardrails.service import GroundedAnswerService
from sre_rag.indexing.artifacts import ChunkArtifactStore
from sre_rag.observability.langfuse import LangfuseTraceProvider
from sre_rag.observability.metrics import PrometheusMetrics
from sre_rag.observability.telemetry import NullWorkflowTelemetry, PipelineTelemetry
from sre_rag.retrieval.bm25 import BM25Retriever
from sre_rag.retrieval.dense import DenseRetriever
from sre_rag.retrieval.embeddings import BGEConfig, BGEEmbedder
from sre_rag.retrieval.fusion import ReciprocalRankFusion
from sre_rag.retrieval.qdrant import QdrantVectorStore
from sre_rag.retrieval.reranking import CrossEncoderConfig, CrossEncoderReranker
from sre_rag.workflow.graph import RAGWorkflow


@dataclass(slots=True)
class RuntimeBundle:
    """Long-lived workflow and resources owned by one API process."""

    workflow: RAGWorkflow
    metrics_registry: CollectorRegistry
    qdrant: QdrantClient
    ollama_http: httpx.Client

    def close(self) -> None:
        self.ollama_http.close()
        self.qdrant.close()


def build_runtime(settings: Settings) -> RuntimeBundle:
    """Load one verified index and construct all query-time components once."""

    if settings.openai_api_key is None:
        raise ValueError("SRE_RAG_OPENAI_API_KEY is required for the primary generator")

    artifact, chunks = ChunkArtifactStore().load_index(
        chunks_path=settings.index_chunks_path,
        metadata_path=settings.index_metadata_path,
    )
    embedder = BGEEmbedder.from_pretrained(BGEConfig(device=settings.model_device))
    qdrant = QdrantClient(url=settings.qdrant_url)
    ollama_http = httpx.Client()
    try:
        vector_store = QdrantVectorStore(
            qdrant,
            collection_name=artifact.dense_collection,
        )
        vector_store.validate_collection(embedder.dimension, expected_points=len(chunks))

        registry = CollectorRegistry()
        telemetry = _telemetry(settings, registry)
        reranker = CrossEncoderReranker.from_pretrained(
            CrossEncoderConfig.bge(device=settings.model_device)
        )
        primary = OpenAIGenerationBackend.from_api_key(
            settings.openai_api_key.get_secret_value(),
            OpenAIConfig(
                model=settings.openai_generation_model,
                timeout_seconds=settings.generation_timeout_seconds,
                reasoning_effort=settings.openai_reasoning_effort,
            ),
        )
        fallback = OllamaGenerationBackend.from_httpx_client(
            ollama_http,
            OllamaConfig(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                timeout_seconds=settings.generation_timeout_seconds,
            ),
        )
        workflow = RAGWorkflow(
            sparse_retriever=BM25Retriever(chunks),
            dense_retriever=DenseRetriever(embedder, vector_store),
            rank_fusion=ReciprocalRankFusion(),
            reranker=reranker,
            answer_service=GroundedAnswerService(
                GroundedDraftGenerator(
                    FallbackGenerationBackend(primary, fallback),
                    GroundedGenerationConfig(
                        max_output_tokens=settings.generation_max_output_tokens
                    ),
                )
            ),
            telemetry=telemetry,
        )
    except BaseException:
        ollama_http.close()
        qdrant.close()
        raise
    return RuntimeBundle(workflow, registry, qdrant, ollama_http)


def _telemetry(
    settings: Settings,
    registry: CollectorRegistry,
) -> PipelineTelemetry | NullWorkflowTelemetry:
    if not settings.langfuse_enabled:
        return NullWorkflowTelemetry()
    return PipelineTelemetry(
        LangfuseTraceProvider.from_environment(),
        PrometheusMetrics(registry),
    )
