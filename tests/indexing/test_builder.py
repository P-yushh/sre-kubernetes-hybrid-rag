"""Tests for composing verified corpora into sparse and dense indexes."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from sre_rag.corpora.fetch import FetchedCorpus
from sre_rag.corpora.manifest import CorpusManifest, CorpusSource
from sre_rag.domain.documents import Chunk, Corpus
from sre_rag.indexing.artifacts import ChunkArtifactStore
from sre_rag.indexing.builder import CorpusIndexBuilder
from sre_rag.ingestion.chunking import HeadingAwareChunker
from sre_rag.ingestion.loaders import MarkdownCorpusLoader
from sre_rag.ingestion.markdown import MarkdownNormalizer
from sre_rag.ingestion.pipeline import IngestionPipeline
from tests.ingestion.helpers import PassThroughConverter, RegexTokenizer


@dataclass
class LocalMaterializer:
    roots: dict[str, Path]

    @contextmanager
    def materialize(self, source: CorpusSource) -> Iterator[FetchedCorpus]:
        root = self.roots[source.name]
        files = tuple(sorted(root.rglob("*.md")))
        yield FetchedCorpus(
            source=source,
            root=root,
            markdown_files=files,
            content_sha256=source.content_sha256,
        )


@dataclass
class RecordingDenseIndexer:
    error: Exception | None = None
    batches: list[tuple[Chunk, ...]] = field(default_factory=list)

    def index(self, chunks: Sequence[Chunk]) -> None:
        if self.error is not None:
            raise self.error
        self.batches.append(tuple(chunks))


def _source(
    name: str,
    corpus: Corpus,
    repository: str,
    include_path: str,
    revision_character: str,
) -> CorpusSource:
    return CorpusSource.model_validate(
        {
            "name": name,
            "corpus": corpus,
            "repository": repository,
            "revision": revision_character * 40,
            "archive_url": (
                f"https://github.com/{repository}/archive/{revision_character * 40}.tar.gz"
            ),
            "content_sha256": revision_character * 64,
            "license": "CC-BY-4.0" if corpus is Corpus.KUBERNETES else "GPL-3.0",
            "license_url": f"https://github.com/{repository}/blob/main/LICENSE",
            "attribution": f"{name} contributors",
            "include_paths": [include_path],
            "expected_markdown_files": 1,
            "max_archive_bytes": 1000,
            "max_extracted_bytes": 1000,
        }
    )


def _fixture(tmp_path: Path) -> tuple[CorpusManifest, Path, dict[str, Path]]:
    kubernetes = _source(
        "kubernetes-docs",
        Corpus.KUBERNETES,
        "kubernetes/website",
        "content/en/docs/tasks",
        "a",
    )
    postmortems = _source(
        "postmortems-app",
        Corpus.POST_MORTEMS,
        "icco/postmortems",
        "data",
        "b",
    )
    kube_root = tmp_path / "kubernetes"
    postmortem_root = tmp_path / "postmortems"
    kube_document = kube_root / "content/en/docs/tasks/debug.md"
    incident_document = postmortem_root / "data/incident.md"
    kube_document.parent.mkdir(parents=True)
    incident_document.parent.mkdir(parents=True)
    kube_document.write_text(
        "---\ntitle: Debug Pods\n---\n# Exit codes\nExit code 137 can indicate OOMKilled.",
        encoding="utf-8",
    )
    incident_document.write_text(
        "---\ntitle: Memory incident\n---\n# Root cause\nA memory limit was exceeded.",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("schema_version: 1\n", encoding="utf-8")
    return (
        CorpusManifest(schema_version=1, sources=(kubernetes, postmortems)),
        manifest_path,
        {"kubernetes-docs": kube_root, "postmortems-app": postmortem_root},
    )


def _builder(roots: dict[str, Path], dense: RecordingDenseIndexer) -> CorpusIndexBuilder:
    return CorpusIndexBuilder(
        LocalMaterializer(roots),
        IngestionPipeline(
            MarkdownCorpusLoader(MarkdownNormalizer(PassThroughConverter())),
            HeadingAwareChunker(RegexTokenizer()),
        ),
        dense,
    )


def test_builder_indexes_same_sorted_chunks_used_by_sparse_artifact(tmp_path: Path) -> None:
    manifest, manifest_path, roots = _fixture(tmp_path)
    dense = RecordingDenseIndexer()
    chunks_path = tmp_path / "output/chunks.jsonl"
    metadata_path = tmp_path / "output/index-manifest.json"

    report = _builder(roots, dense).build(
        manifest,
        manifest_path=manifest_path,
        chunks_path=chunks_path,
        metadata_path=metadata_path,
        dense_collection="sre_kubernetes_chunks_test",
    )
    loaded = ChunkArtifactStore().load(
        chunks_path=chunks_path,
        metadata_path=metadata_path,
    )

    assert report.artifact.chunk_count == 2
    assert [source.document_count for source in report.artifact.sources] == [1, 1]
    assert dense.batches == [loaded]
    assert [chunk.chunk_id for chunk in loaded] == sorted(chunk.chunk_id for chunk in loaded)
    assert {chunk.provenance.license for chunk in loaded} == {"CC-BY-4.0", "GPL-3.0"}
    assert all(chunk.provenance.revision in {"a" * 40, "b" * 40} for chunk in loaded)


def test_builder_does_not_publish_artifacts_when_dense_indexing_fails(tmp_path: Path) -> None:
    manifest, manifest_path, roots = _fixture(tmp_path)
    chunks_path = tmp_path / "output/chunks.jsonl"
    metadata_path = tmp_path / "output/index-manifest.json"

    with pytest.raises(RuntimeError, match="Qdrant unavailable"):
        _builder(roots, RecordingDenseIndexer(RuntimeError("Qdrant unavailable"))).build(
            manifest,
            manifest_path=manifest_path,
            chunks_path=chunks_path,
            metadata_path=metadata_path,
            dense_collection="sre_kubernetes_chunks_test",
        )

    assert not chunks_path.exists()
    assert not metadata_path.exists()


def test_builder_rejects_source_without_indexable_documents(tmp_path: Path) -> None:
    source = _source(
        "kubernetes-docs",
        Corpus.KUBERNETES,
        "kubernetes/website",
        "content/en/docs/tasks",
        "a",
    )
    empty_root = tmp_path / "empty"
    (empty_root / "content/en/docs/tasks").mkdir(parents=True)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("schema_version: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no indexable content"):
        _builder({source.name: empty_root}, RecordingDenseIndexer()).build(
            CorpusManifest(schema_version=1, sources=(source,)),
            manifest_path=manifest_path,
            chunks_path=tmp_path / "chunks.jsonl",
            metadata_path=tmp_path / "metadata.json",
            dense_collection="sre_kubernetes_chunks_test",
        )
