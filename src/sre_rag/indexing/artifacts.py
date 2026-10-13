"""Deterministic, fail-closed chunk artifacts used to rebuild sparse retrieval."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from pydantic import Field, model_validator

from sre_rag.domain.base import DomainModel, NonEmptyText, Revision, Sha256
from sre_rag.domain.documents import Chunk, Corpus


class IndexArtifactError(RuntimeError):
    """Raised when an index artifact is incomplete, corrupt, or contradictory."""


class IndexedSource(DomainModel):
    """Counts and provenance retained for one indexed source snapshot."""

    name: NonEmptyText
    corpus: Corpus
    revision: Revision
    content_sha256: Sha256
    license: NonEmptyText
    document_count: int = Field(gt=0)
    chunk_count: int = Field(gt=0)


class IndexArtifact(DomainModel):
    """Machine-readable identity of one complete sparse/dense index build."""

    schema_version: Literal[1] = 1
    corpus_manifest_sha256: Sha256
    dense_collection: NonEmptyText
    chunks_file: NonEmptyText
    chunks_sha256: Sha256
    chunk_count: int = Field(gt=0)
    sources: tuple[IndexedSource, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def source_counts_must_match_total(self) -> IndexArtifact:
        if sum(source.chunk_count for source in self.sources) != self.chunk_count:
            raise ValueError("source chunk counts must equal the artifact chunk count")
        return self


class ChunkArtifactStore:
    """Write and load canonical JSONL chunks with independent metadata checksums."""

    def write(
        self,
        chunks: Sequence[Chunk],
        *,
        chunks_path: Path,
        metadata_path: Path,
        corpus_manifest_sha256: str,
        dense_collection: str,
        sources: Sequence[IndexedSource],
    ) -> IndexArtifact:
        ordered = tuple(sorted(chunks, key=lambda chunk: chunk.chunk_id))
        _require_unique_chunk_ids(ordered)
        if not ordered:
            raise ValueError("cannot write an empty chunk artifact")

        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_temporary = _temporary_path(chunks_path)
        metadata_temporary = _temporary_path(metadata_path)
        try:
            digest = hashlib.sha256()
            with chunks_temporary.open("wb") as output:
                for chunk in ordered:
                    line = _json_line(chunk.model_dump(mode="json"))
                    digest.update(line)
                    output.write(line)

            artifact = IndexArtifact(
                corpus_manifest_sha256=corpus_manifest_sha256,
                dense_collection=dense_collection,
                chunks_file=chunks_path.name,
                chunks_sha256=digest.hexdigest(),
                chunk_count=len(ordered),
                sources=tuple(sources),
            )
            metadata_temporary.write_text(
                json.dumps(
                    artifact.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            chunks_temporary.replace(chunks_path)
            metadata_temporary.replace(metadata_path)
            return artifact
        finally:
            chunks_temporary.unlink(missing_ok=True)
            metadata_temporary.unlink(missing_ok=True)

    def load(self, *, chunks_path: Path, metadata_path: Path) -> tuple[Chunk, ...]:
        """Load and validate chunks while hiding storage metadata from callers."""

        _, chunks = self.load_index(chunks_path=chunks_path, metadata_path=metadata_path)
        return chunks

    def load_index(
        self,
        *,
        chunks_path: Path,
        metadata_path: Path,
    ) -> tuple[IndexArtifact, tuple[Chunk, ...]]:
        """Load a complete artifact and fail closed on any contradiction."""

        try:
            artifact = IndexArtifact.model_validate_json(metadata_path.read_text(encoding="utf-8"))
            chunks_bytes = chunks_path.read_bytes()
        except (OSError, ValueError) as error:
            raise IndexArtifactError("index artifact files are missing or invalid") from error

        if artifact.chunks_file != chunks_path.name:
            raise IndexArtifactError("index metadata references a different chunk file")
        if hashlib.sha256(chunks_bytes).hexdigest() != artifact.chunks_sha256:
            raise IndexArtifactError("chunk artifact checksum does not match its metadata")

        chunks: list[Chunk] = []
        try:
            for line in chunks_bytes.splitlines():
                if line.strip():
                    chunks.append(Chunk.model_validate_json(line))
        except ValueError as error:
            raise IndexArtifactError("chunk artifact contains invalid JSONL") from error
        if len(chunks) != artifact.chunk_count:
            raise IndexArtifactError("chunk artifact count does not match its metadata")
        try:
            _require_unique_chunk_ids(chunks)
        except ValueError as error:
            raise IndexArtifactError("chunk artifact contains duplicate chunk IDs") from error
        return artifact, tuple(chunks)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _temporary_path(target: Path) -> Path:
    with NamedTemporaryFile(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    ) as temporary:
        return Path(temporary.name)


def _json_line(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _require_unique_chunk_ids(chunks: Sequence[Chunk]) -> None:
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("chunk artifact requires unique chunk IDs")
