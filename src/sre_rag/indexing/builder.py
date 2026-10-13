"""Composition of verified corpus acquisition, ingestion, and hybrid indexing."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sre_rag.corpora.fetch import FetchedCorpus
from sre_rag.corpora.manifest import CorpusManifest, CorpusSource
from sre_rag.domain.documents import Chunk
from sre_rag.indexing.artifacts import (
    ChunkArtifactStore,
    IndexArtifact,
    IndexedSource,
    sha256_file,
)
from sre_rag.ingestion.loaders import CorpusConfig
from sre_rag.ingestion.pipeline import IngestionPipeline


class CorpusMaterializer(Protocol):
    def materialize(self, source: CorpusSource) -> AbstractContextManager[FetchedCorpus]: ...


class DenseIndexer(Protocol):
    def index(self, chunks: Sequence[Chunk]) -> None: ...


@dataclass(frozen=True, slots=True)
class IndexBuildReport:
    """Evidence that one corpus revision was fully transformed and indexed."""

    artifact: IndexArtifact
    chunks_path: Path
    metadata_path: Path


class CorpusIndexBuilder:
    """Build dense and sparse inputs from the same verified immutable chunks."""

    def __init__(
        self,
        materializer: CorpusMaterializer,
        ingestion: IngestionPipeline,
        dense_indexer: DenseIndexer,
        artifact_store: ChunkArtifactStore | None = None,
    ) -> None:
        self._materializer = materializer
        self._ingestion = ingestion
        self._dense_indexer = dense_indexer
        self._artifact_store = artifact_store or ChunkArtifactStore()

    def build(
        self,
        manifest: CorpusManifest,
        *,
        manifest_path: Path,
        chunks_path: Path,
        metadata_path: Path,
        dense_collection: str,
    ) -> IndexBuildReport:
        chunks: list[Chunk] = []
        summaries: list[IndexedSource] = []
        for source, corpus in self._materialized(manifest.sources):
            batch = self._ingestion.run(
                CorpusConfig(
                    root=corpus.root,
                    corpus=source.corpus,
                    repository=source.repository,
                    revision=source.revision,
                    license=source.license,
                    include_paths=source.include_paths,
                )
            )
            if not batch.documents or not batch.chunks:
                raise ValueError(f"source {source.name!r} produced no indexable content")
            chunks.extend(batch.chunks)
            summaries.append(
                IndexedSource(
                    name=source.name,
                    corpus=source.corpus,
                    revision=source.revision,
                    content_sha256=source.content_sha256,
                    license=source.license,
                    document_count=len(batch.documents),
                    chunk_count=len(batch.chunks),
                )
            )

        ordered = tuple(sorted(chunks, key=lambda chunk: chunk.chunk_id))
        chunk_ids = [chunk.chunk_id for chunk in ordered]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("index build produced duplicate chunk IDs")
        self._dense_indexer.index(ordered)
        artifact = self._artifact_store.write(
            ordered,
            chunks_path=chunks_path,
            metadata_path=metadata_path,
            corpus_manifest_sha256=sha256_file(manifest_path),
            dense_collection=dense_collection,
            sources=summaries,
        )
        return IndexBuildReport(
            artifact=artifact,
            chunks_path=chunks_path,
            metadata_path=metadata_path,
        )

    def _materialized(
        self,
        sources: Sequence[CorpusSource],
    ) -> Iterator[tuple[CorpusSource, FetchedCorpus]]:
        for source in sources:
            with self._materializer.materialize(source) as corpus:
                yield source, corpus
