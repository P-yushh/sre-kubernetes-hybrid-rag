"""Tests for the pinned-corpus verification command."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

import sre_rag.corpora.cli as cli
from sre_rag.corpora.fetch import FetchedCorpus
from sre_rag.corpora.manifest import CorpusManifest, CorpusSource
from sre_rag.domain.documents import Corpus


def _source() -> CorpusSource:
    return CorpusSource(
        name="kubernetes-docs",
        corpus=Corpus.KUBERNETES,
        repository="kubernetes/website",
        revision="a" * 40,
        archive_url=f"https://github.com/kubernetes/website/archive/{'a' * 40}.tar.gz",
        content_sha256="b" * 64,
        license="CC-BY-4.0",
        license_url="https://github.com/kubernetes/website/blob/main/LICENSE",
        attribution="Kubernetes contributors",
        include_paths=("content/en/docs/tasks",),
        expected_markdown_files=1,
        max_archive_bytes=1000,
        max_extracted_bytes=1000,
    )


def test_verify_command_reports_snapshot_and_closes_downloader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source()
    markdown = tmp_path / "content/en/docs/tasks/debug.md"
    markdown.parent.mkdir(parents=True)
    markdown.write_text("# Debug", encoding="utf-8")
    closed = False

    class FakeDownloader:
        def close(self) -> None:
            nonlocal closed
            closed = True

    class FakeFetcher:
        def __init__(self, downloader: FakeDownloader) -> None:
            del downloader

        @contextmanager
        def materialize(self, selected: CorpusSource) -> Iterator[FetchedCorpus]:
            yield FetchedCorpus(
                source=selected,
                root=tmp_path,
                markdown_files=(markdown,),
                content_sha256=selected.content_sha256,
            )

    monkeypatch.setattr(
        cli,
        "load_manifest",
        lambda path: CorpusManifest(schema_version=1, sources=(source,)),
    )
    monkeypatch.setattr(cli, "HttpxArchiveDownloader", FakeDownloader)
    monkeypatch.setattr(cli, "CorpusFetcher", FakeFetcher)

    assert cli.main(["--manifest", str(tmp_path / "manifest.yaml")]) == 0
    assert closed is True
    assert "kubernetes-docs: 1 Markdown files verified" in capsys.readouterr().out
