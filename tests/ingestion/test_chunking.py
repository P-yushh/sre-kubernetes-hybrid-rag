"""Tests for heading-aware token-window chunking."""

from pathlib import PurePosixPath

import pytest

from sre_rag.domain.documents import Corpus, SourceDocument, SourceProvenance
from sre_rag.ingestion.chunking import ChunkingConfig, HeadingAwareChunker
from sre_rag.ingestion.identifiers import sha256_text
from tests.ingestion.helpers import RegexTokenizer

REVISION = "d" * 40


def make_document(content: str) -> SourceDocument:
    return SourceDocument(
        document_id="kubernetes:content/en/docs/tasks/debug/pods.md",
        title="Debug Pods",
        content=content,
        content_sha256=sha256_text(content),
        provenance=SourceProvenance(
            corpus=Corpus.KUBERNETES,
            repository="kubernetes/website",
            revision=REVISION,
            relative_path=PurePosixPath("content/en/docs/tasks/debug/pods.md").as_posix(),
            source_url=f"https://github.com/kubernetes/website/blob/{REVISION}/pods.md",
            license="CC-BY-4.0",
        ),
    )


def test_chunker_is_deterministic_and_preserves_heading_context() -> None:
    document = make_document(
        "# Pod failures\nExit code 137 means forced termination.\n\n"
        "Inspect CrashLoopBackOff carefully.\n## Configuration\n"
        "Use --enable-admission-plugins exactly."
    )
    chunker = HeadingAwareChunker(RegexTokenizer(), ChunkingConfig(max_tokens=8, overlap_tokens=2))

    first_run = chunker.chunk(document)
    second_run = chunker.chunk(document)

    assert first_run == second_run
    assert first_run[0].heading_path == ("Pod failures",)
    assert first_run[-1].heading_path == ("Pod failures", "Configuration")
    assert first_run[0].chunk_id.endswith("#pod-failures@0")
    assert "137" in first_run[0].text
    assert "--enable-admission-plugins" in first_run[-1].text
    assert all(chunk.token_count <= 8 for chunk in first_run)


def test_oversized_block_uses_exact_token_overlap() -> None:
    document = make_document("# Numbers\none two three four five six seven eight nine ten")
    chunker = HeadingAwareChunker(RegexTokenizer(), ChunkingConfig(max_tokens=4, overlap_tokens=1))

    chunks = chunker.chunk(document)

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
        "seven eight nine ten",
    ]


def test_fenced_yaml_is_not_split_when_it_fits() -> None:
    yaml_block = "```yaml\nmaxSurge: 1\nmaxUnavailable: 0\n```"
    document = make_document(f"# Deployment\n{yaml_block}")
    chunker = HeadingAwareChunker(RegexTokenizer(), ChunkingConfig(max_tokens=10, overlap_tokens=2))

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks[0].text == yaml_block


def test_code_block_integrity_takes_priority_when_overlap_does_not_fit() -> None:
    document = make_document(
        "# Lifecycle\none two three four\n\nfive six seven four\n\n"
        "```bash\nkubectl get pods\n```\n\nnine ten"
    )
    chunker = HeadingAwareChunker(RegexTokenizer(), ChunkingConfig(max_tokens=8, overlap_tokens=4))

    chunks = chunker.chunk(document)

    assert "four" in chunks[0].text
    assert sum("kubectl get pods" in chunk.text for chunk in chunks) == 1
    assert any("```bash\nkubectl get pods\n```" in chunk.text for chunk in chunks)


@pytest.mark.parametrize(("max_tokens", "overlap_tokens"), [(0, 0), (10, -1), (10, 10), (10, 11)])
def test_chunking_config_rejects_invalid_windows(max_tokens: int, overlap_tokens: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
