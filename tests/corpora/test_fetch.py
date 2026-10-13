"""Tests for bounded download, extraction safety, and ephemeral cleanup."""

import io
import tarfile
from pathlib import Path

import httpx
import pytest

from sre_rag.corpora.fetch import (
    CorpusFetcher,
    CorpusFetchError,
    HttpxArchiveDownloader,
    content_sha256,
)
from sre_rag.corpora.manifest import CorpusSource
from sre_rag.domain.documents import Corpus


class BytesDownloader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.destinations: list[Path] = []

    def download(self, url: str, destination: Path, *, max_bytes: int) -> None:
        del url
        if len(self.payload) > max_bytes:
            raise CorpusFetchError("fixture exceeds limit")
        destination.write_bytes(self.payload)
        self.destinations.append(destination)


def _source(digest: str, **updates: object) -> CorpusSource:
    values: dict[str, object] = {
        "name": "test-corpus",
        "corpus": Corpus.KUBERNETES,
        "repository": "example/corpus",
        "revision": "a" * 40,
        "archive_url": f"https://github.com/example/corpus/archive/{'a' * 40}.tar.gz",
        "content_sha256": digest,
        "license": "CC-BY-4.0",
        "license_url": "https://example.com/license",
        "attribution": "Example contributors",
        "include_paths": ("docs",),
        "expected_markdown_files": 2,
        "max_archive_bytes": 10_000,
        "max_extracted_bytes": 1_000,
    }
    values.update(updates)
    return CorpusSource.model_validate(values)


def _archive(
    files: dict[str, bytes],
    *,
    special: tarfile.TarInfo | None = None,
) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name=f"repository-revision/{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        if special is not None:
            archive.addfile(special)
    return payload.getvalue()


def _expected_digest(tmp_path: Path) -> str:
    root = tmp_path / "expected"
    (root / "docs").mkdir(parents=True)
    (root / "docs/first.md").write_text("first", encoding="utf-8")
    (root / "docs/second.md").write_text("second", encoding="utf-8")
    return content_sha256(root)


def test_fetcher_selects_markdown_verifies_digest_and_deletes_snapshot(tmp_path: Path) -> None:
    payload = _archive(
        {
            "docs/first.md": b"first",
            "docs/second.md": b"second",
            "docs/._second.md": b"macOS metadata",
            "docs/ignored.txt": b"ignored",
            "outside/not-selected.md": b"outside",
        }
    )
    downloader = BytesDownloader(payload)
    materialized_root: Path | None = None

    with CorpusFetcher(downloader).materialize(_source(_expected_digest(tmp_path))) as corpus:
        materialized_root = corpus.root
        assert materialized_root.exists()
        assert [path.name for path in corpus.markdown_files] == ["first.md", "second.md"]
        assert not (materialized_root / "docs/ignored.txt").exists()
        assert not (materialized_root / "outside/not-selected.md").exists()

    assert materialized_root is not None
    assert not materialized_root.exists()
    assert not downloader.destinations[0].exists()


def test_fetcher_rejects_checksum_and_file_count_mismatches(tmp_path: Path) -> None:
    payload = _archive({"docs/first.md": b"first", "docs/second.md": b"second"})
    fetcher = CorpusFetcher(BytesDownloader(payload))

    with pytest.raises(CorpusFetchError, match="expected 3 Markdown files"):
        with fetcher.materialize(_source(_expected_digest(tmp_path), expected_markdown_files=3)):
            pytest.fail("invalid file count must fail before yielding")

    with pytest.raises(CorpusFetchError, match="checksum"):
        with fetcher.materialize(_source("0" * 64)):
            pytest.fail("invalid checksum must fail before yielding")


@pytest.mark.parametrize(
    "special",
    [
        tarfile.TarInfo(name="repository-revision/docs/../../escape.md"),
        tarfile.TarInfo(name="second-root/docs/other.md"),
    ],
)
def test_fetcher_rejects_unsafe_or_multiple_archive_roots(
    tmp_path: Path,
    special: tarfile.TarInfo,
) -> None:
    payload = _archive(
        {"docs/first.md": b"first", "docs/second.md": b"second"},
        special=special,
    )

    with pytest.raises(CorpusFetchError):
        with CorpusFetcher(BytesDownloader(payload)).materialize(
            _source(_expected_digest(tmp_path))
        ):
            pytest.fail("unsafe archive must fail before yielding")


def test_fetcher_rejects_selected_symbolic_link(tmp_path: Path) -> None:
    link = tarfile.TarInfo(name="repository-revision/docs/link.md")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../outside"
    payload = _archive(
        {"docs/first.md": b"first", "docs/second.md": b"second"},
        special=link,
    )

    with pytest.raises(CorpusFetchError, match="non-regular"):
        with CorpusFetcher(BytesDownloader(payload)).materialize(
            _source(_expected_digest(tmp_path))
        ):
            pytest.fail("selected links must fail before yielding")


def test_fetcher_enforces_extracted_size_limit(tmp_path: Path) -> None:
    payload = _archive({"docs/first.md": b"first", "docs/second.md": b"second"})

    with pytest.raises(CorpusFetchError, match="exceeds"):
        with CorpusFetcher(BytesDownloader(payload)).materialize(
            _source(_expected_digest(tmp_path), max_extracted_bytes=5)
        ):
            pytest.fail("oversized corpus must fail before yielding")


def test_http_downloader_streams_to_disk_and_enforces_size(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"archive"))
    client = httpx.Client(transport=transport)
    downloader = HttpxArchiveDownloader(client)

    with client:
        destination = tmp_path / "archive.tar.gz"
        downloader.download("https://example.com/archive", destination, max_bytes=7)
        assert destination.read_bytes() == b"archive"
        with pytest.raises(CorpusFetchError, match="exceeds"):
            downloader.download("https://example.com/archive", destination, max_bytes=6)


def test_http_downloader_normalizes_transport_failures(tmp_path: Path) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = httpx.Client(transport=httpx.MockTransport(fail))
    downloader = HttpxArchiveDownloader(client)

    with client, pytest.raises(CorpusFetchError, match="download failed"):
        downloader.download("https://example.com/archive", tmp_path / "archive", max_bytes=10)


@pytest.mark.parametrize(("timeout", "max_bytes"), [(0, 1), (1, 0)])
def test_http_downloader_rejects_invalid_limits(
    tmp_path: Path,
    timeout: float,
    max_bytes: int,
) -> None:
    if timeout == 0:
        with pytest.raises(ValueError, match="timeout"):
            HttpxArchiveDownloader(timeout_seconds=timeout)
        return

    downloader = HttpxArchiveDownloader(timeout_seconds=timeout)
    try:
        with pytest.raises(ValueError, match="archive size"):
            downloader.download("https://example.com", tmp_path / "archive", max_bytes=max_bytes)
    finally:
        downloader.close()
