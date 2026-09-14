"""Tests for ingestion component composition."""

from pathlib import Path, PurePosixPath

from sre_rag.domain.documents import Corpus
from sre_rag.ingestion.chunking import ChunkingConfig, HeadingAwareChunker
from sre_rag.ingestion.loaders import CorpusConfig, MarkdownCorpusLoader
from sre_rag.ingestion.markdown import MarkdownNormalizer
from sre_rag.ingestion.pipeline import IngestionPipeline
from tests.ingestion.helpers import PassThroughConverter, RegexTokenizer


def test_pipeline_loads_and_chunks_a_versioned_corpus(tmp_path: Path) -> None:
    source = tmp_path / "content/en/docs/tasks/debug-pods.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "---\ntitle: Debug Pods\n---\n# Exit codes\nExit code 137 can indicate OOMKilled.",
        encoding="utf-8",
    )
    config = CorpusConfig(
        root=tmp_path,
        corpus=Corpus.KUBERNETES,
        repository="kubernetes/website",
        revision="e" * 40,
        license="CC-BY-4.0",
        include_paths=(PurePosixPath("content/en/docs/tasks"),),
    )
    pipeline = IngestionPipeline(
        MarkdownCorpusLoader(MarkdownNormalizer(PassThroughConverter())),
        HeadingAwareChunker(RegexTokenizer(), ChunkingConfig(max_tokens=16, overlap_tokens=2)),
    )

    batch = pipeline.run(config)

    assert len(batch.documents) == 1
    assert len(batch.chunks) == 1
    assert batch.chunks[0].document_id == batch.documents[0].document_id
    assert batch.chunks[0].text == "Exit code 137 can indicate OOMKilled."
