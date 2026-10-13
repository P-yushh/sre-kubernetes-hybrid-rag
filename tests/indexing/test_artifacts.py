"""Tests for deterministic, self-validating chunk artifacts."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sre_rag.domain.documents import Chunk, Corpus
from sre_rag.indexing.artifacts import (
    ChunkArtifactStore,
    IndexArtifact,
    IndexArtifactError,
    IndexedSource,
)
from tests.domain.factories import make_chunk

_DENSE_COLLECTION = "sre_kubernetes_chunks_aaaaaaaaaaaa"


def _source(chunk_count: int = 2) -> IndexedSource:
    return IndexedSource(
        name="kubernetes-docs",
        corpus=Corpus.KUBERNETES,
        revision="a" * 40,
        content_sha256="b" * 64,
        license="CC-BY-4.0",
        document_count=1,
        chunk_count=chunk_count,
    )


def _chunks() -> tuple[Chunk, Chunk]:
    first = make_chunk().model_copy(update={"chunk_id": "chunk-b"})
    second = make_chunk().model_copy(update={"chunk_id": "chunk-a", "ordinal": 1})
    return first, second


def test_artifact_round_trip_is_sorted_and_byte_deterministic(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    metadata_path = tmp_path / "index-manifest.json"
    store = ChunkArtifactStore()
    chunks = _chunks()

    artifact = store.write(
        chunks,
        chunks_path=chunks_path,
        metadata_path=metadata_path,
        corpus_manifest_sha256="c" * 64,
        dense_collection=_DENSE_COLLECTION,
        sources=(_source(),),
    )
    first_chunks_bytes = chunks_path.read_bytes()
    first_metadata_bytes = metadata_path.read_bytes()
    loaded = store.load(chunks_path=chunks_path, metadata_path=metadata_path)

    assert [chunk.chunk_id for chunk in loaded] == ["chunk-a", "chunk-b"]
    assert artifact.chunks_sha256 == hashlib.sha256(first_chunks_bytes).hexdigest()

    store.write(
        tuple(reversed(chunks)),
        chunks_path=chunks_path,
        metadata_path=metadata_path,
        corpus_manifest_sha256="c" * 64,
        dense_collection=_DENSE_COLLECTION,
        sources=(_source(),),
    )
    assert chunks_path.read_bytes() == first_chunks_bytes
    assert metadata_path.read_bytes() == first_metadata_bytes


def test_artifact_rejects_empty_and_duplicate_chunks(tmp_path: Path) -> None:
    store = ChunkArtifactStore()
    chunks_path = tmp_path / "chunks.jsonl"
    metadata_path = tmp_path / "metadata.json"

    with pytest.raises(ValueError, match="empty"):
        store.write(
            (),
            chunks_path=chunks_path,
            metadata_path=metadata_path,
            corpus_manifest_sha256="c" * 64,
            dense_collection=_DENSE_COLLECTION,
            sources=(_source(),),
        )

    chunk = make_chunk()
    with pytest.raises(ValueError, match="unique"):
        store.write(
            (chunk, chunk),
            chunks_path=chunks_path,
            metadata_path=metadata_path,
            corpus_manifest_sha256="c" * 64,
            dense_collection=_DENSE_COLLECTION,
            sources=(_source(),),
        )


def test_artifact_load_fails_closed_on_tampering(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    metadata_path = tmp_path / "metadata.json"
    store = ChunkArtifactStore()
    store.write(
        _chunks(),
        chunks_path=chunks_path,
        metadata_path=metadata_path,
        corpus_manifest_sha256="c" * 64,
        dense_collection=_DENSE_COLLECTION,
        sources=(_source(),),
    )
    chunks_path.write_bytes(chunks_path.read_bytes() + b"\n")

    with pytest.raises(IndexArtifactError, match="checksum"):
        store.load(chunks_path=chunks_path, metadata_path=metadata_path)


def test_artifact_load_rejects_wrong_file_reference_and_invalid_jsonl(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    metadata_path = tmp_path / "metadata.json"
    store = ChunkArtifactStore()
    artifact = store.write(
        _chunks(),
        chunks_path=chunks_path,
        metadata_path=metadata_path,
        corpus_manifest_sha256="c" * 64,
        dense_collection=_DENSE_COLLECTION,
        sources=(_source(),),
    )
    metadata_path.write_text(
        artifact.model_copy(update={"chunks_file": "other.jsonl"}).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(IndexArtifactError, match="different chunk file"):
        store.load(chunks_path=chunks_path, metadata_path=metadata_path)

    invalid = b"not-json\n"
    metadata_path.write_text(
        artifact.model_copy(
            update={
                "chunks_sha256": hashlib.sha256(invalid).hexdigest(),
                "chunk_count": 1,
                "sources": (_source(chunk_count=1),),
            }
        ).model_dump_json(),
        encoding="utf-8",
    )
    chunks_path.write_bytes(invalid)
    with pytest.raises(IndexArtifactError, match="invalid JSONL"):
        store.load(chunks_path=chunks_path, metadata_path=metadata_path)


def test_artifact_contract_rejects_contradictory_source_counts() -> None:
    with pytest.raises(ValidationError, match="source chunk counts"):
        IndexArtifact(
            corpus_manifest_sha256="a" * 64,
            dense_collection=_DENSE_COLLECTION,
            chunks_file="chunks.jsonl",
            chunks_sha256="b" * 64,
            chunk_count=3,
            sources=(_source(chunk_count=2),),
        )


def test_load_normalizes_missing_or_invalid_metadata(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    metadata = tmp_path / "metadata.json"
    chunks.write_text("", encoding="utf-8")
    metadata.write_text(json.dumps({"invalid": True}), encoding="utf-8")

    with pytest.raises(IndexArtifactError, match="missing or invalid"):
        ChunkArtifactStore().load(chunks_path=chunks, metadata_path=metadata)
