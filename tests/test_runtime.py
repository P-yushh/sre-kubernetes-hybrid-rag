"""Tests for fail-closed production runtime composition and ownership."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

import sre_rag.runtime as runtime_module
from sre_rag.config import Settings
from sre_rag.domain.documents import Chunk, Corpus
from sre_rag.domain.generation import GenerationResult
from sre_rag.generation.base import GenerationRequest
from sre_rag.generation.ollama import OllamaGenerationBackend
from sre_rag.generation.openai import OpenAIGenerationBackend
from sre_rag.indexing.artifacts import ChunkArtifactStore, IndexedSource
from sre_rag.main import create_runtime_app
from sre_rag.retrieval.embeddings import BGEEmbedder, Vector
from sre_rag.retrieval.qdrant import QdrantVectorStore
from sre_rag.retrieval.reranking import CrossEncoderReranker
from tests.domain.factories import make_chunk
from tests.generation.helpers import make_generation_result


class FakeEmbedder:
    dimension = 3

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        return tuple((1.0, 0.0, 0.0) for _ in texts)

    def embed_query(self, text: str) -> Vector:
        del text
        return (1.0, 0.0, 0.0)


class FakeBackend:
    def generate(self, request: GenerationRequest) -> GenerationResult:
        del request
        return make_generation_result()


class FakeReranker:
    def rerank(self, query: object, candidates: object, *, limit: int | None = None) -> tuple[()]:
        del query, candidates, limit
        return ()


def _write_artifact(tmp_path: Path, chunk: Chunk) -> tuple[Path, Path]:
    chunks_path = tmp_path / "chunks.jsonl"
    metadata_path = tmp_path / "index-manifest.json"
    ChunkArtifactStore().write(
        (chunk,),
        chunks_path=chunks_path,
        metadata_path=metadata_path,
        corpus_manifest_sha256="a" * 64,
        dense_collection="chunks_aaaaaaaaaaaa",
        sources=(
            IndexedSource(
                name="kubernetes-docs",
                corpus=Corpus.KUBERNETES,
                revision="b" * 40,
                content_sha256="c" * 64,
                license="CC-BY-4.0",
                document_count=1,
                chunk_count=1,
            ),
        ),
    )
    return chunks_path, metadata_path


def test_build_runtime_loads_matching_versioned_dense_and_sparse_indexes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = make_chunk()
    chunks_path, metadata_path = _write_artifact(tmp_path, chunk)
    qdrant = QdrantClient(":memory:")
    dense = QdrantVectorStore(qdrant, collection_name="chunks_aaaaaaaaaaaa")
    dense.ensure_collection(3)
    dense.upsert((chunk,), ((1.0, 0.0, 0.0),))

    monkeypatch.setattr(runtime_module, "QdrantClient", lambda **kwargs: qdrant)
    monkeypatch.setattr(
        BGEEmbedder,
        "from_pretrained",
        lambda config: FakeEmbedder(),
    )
    monkeypatch.setattr(
        CrossEncoderReranker,
        "from_pretrained",
        lambda config: FakeReranker(),
    )
    monkeypatch.setattr(
        OpenAIGenerationBackend,
        "from_api_key",
        lambda api_key, config: FakeBackend(),
    )
    monkeypatch.setattr(
        OllamaGenerationBackend,
        "from_httpx_client",
        lambda client, config: FakeBackend(),
    )
    settings = Settings(
        environment="test",
        openai_api_key="secret",
        index_chunks_path=chunks_path,
        index_metadata_path=metadata_path,
        langfuse_enabled=False,
    )

    bundle = runtime_module.build_runtime(settings)

    assert bundle.workflow is not None
    assert bundle.metrics_registry is not None
    bundle.close()


def test_build_runtime_requires_primary_provider_credentials() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        runtime_module.build_runtime(Settings(environment="test", openai_api_key=None))


def test_runtime_app_owns_bundle_during_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    closed = False

    class FakeBundle:
        workflow = object()
        metrics_registry = object()

        def close(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr(runtime_module, "build_runtime", lambda settings: FakeBundle())
    application = create_runtime_app(Settings(environment="test"))

    with TestClient(application) as client:
        assert client.get("/readyz").json() == {"status": "ready"}

    assert closed is True
