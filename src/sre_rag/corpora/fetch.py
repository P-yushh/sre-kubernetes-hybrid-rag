"""Bounded download and safe temporary extraction of pinned corpus archives."""

from __future__ import annotations

import hashlib
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import IO, Protocol

import httpx

from sre_rag.corpora.manifest import CorpusSource

_COPY_BLOCK_SIZE = 1024 * 1024


class CorpusFetchError(RuntimeError):
    """Raised when a remote corpus cannot be materialized safely and exactly."""


class ArchiveDownloader(Protocol):
    """Download one remote archive into a caller-owned temporary file."""

    def download(self, url: str, destination: Path, *, max_bytes: int) -> None: ...


class HttpxArchiveDownloader:
    """Stream archives with redirects, timeouts, and a strict byte ceiling."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        timeout_seconds: float = 120.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("download timeout must be positive")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(timeout_seconds),
            headers={"User-Agent": "sre-kubernetes-hybrid-rag/0.1"},
        )

    def download(self, url: str, destination: Path, *, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("maximum archive size must be positive")
        try:
            with self._client.stream("GET", url) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length is not None and int(content_length) > max_bytes:
                    raise CorpusFetchError("remote archive exceeds the configured size limit")

                downloaded = 0
                with destination.open("wb") as archive:
                    for block in response.iter_bytes(_COPY_BLOCK_SIZE):
                        downloaded += len(block)
                        if downloaded > max_bytes:
                            raise CorpusFetchError(
                                "downloaded archive exceeds the configured size limit"
                            )
                        archive.write(block)
        except (httpx.HTTPError, ValueError) as error:
            if isinstance(error, CorpusFetchError):
                raise
            raise CorpusFetchError("corpus archive download failed") from error

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


@dataclass(frozen=True, slots=True)
class FetchedCorpus:
    """Verified corpus material that exists only within the fetch context."""

    source: CorpusSource
    root: Path
    markdown_files: tuple[Path, ...]
    content_sha256: str


class CorpusFetcher:
    """Materialize selected Markdown from an immutable archive, then delete it."""

    def __init__(self, downloader: ArchiveDownloader) -> None:
        self._downloader = downloader

    @contextmanager
    def materialize(self, source: CorpusSource) -> Iterator[FetchedCorpus]:
        with TemporaryDirectory(prefix=f"sre-rag-{source.name}-") as temporary:
            temporary_root = Path(temporary)
            archive_path = temporary_root / "corpus.tar.gz"
            corpus_root = temporary_root / "corpus"
            corpus_root.mkdir()
            self._downloader.download(
                str(source.archive_url),
                archive_path,
                max_bytes=source.max_archive_bytes,
            )
            _extract_selected_markdown(
                archive_path,
                corpus_root,
                source.include_paths,
                max_extracted_bytes=source.max_extracted_bytes,
            )
            markdown_files = _markdown_files(corpus_root)
            if len(markdown_files) != source.expected_markdown_files:
                raise CorpusFetchError(
                    f"{source.name} expected {source.expected_markdown_files} Markdown files, "
                    f"received {len(markdown_files)}"
                )
            digest = content_sha256(corpus_root)
            if digest != source.content_sha256:
                raise CorpusFetchError(
                    f"{source.name} content checksum does not match its pinned manifest"
                )
            yield FetchedCorpus(
                source=source,
                root=corpus_root,
                markdown_files=markdown_files,
                content_sha256=digest,
            )


def content_sha256(root: Path) -> str:
    """Hash relative paths and bytes for a compression-independent corpus identity."""

    digest = hashlib.sha256()
    for path in _markdown_files(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as document:
            while block := document.read(_COPY_BLOCK_SIZE):
                digest.update(block)
    return digest.hexdigest()


def _extract_selected_markdown(
    archive_path: Path,
    destination: Path,
    include_paths: tuple[PurePosixPath, ...],
    *,
    max_extracted_bytes: int,
) -> None:
    extracted_bytes = 0
    archive_root: str | None = None
    extracted_paths: set[PurePosixPath] = set()
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                path = _safe_archive_path(member.name)
                if len(path.parts) < 2:
                    continue
                member_root = path.parts[0]
                if archive_root is None:
                    archive_root = member_root
                elif member_root != archive_root:
                    raise CorpusFetchError("corpus archive contains multiple top-level roots")
                relative = PurePosixPath(*path.parts[1:])
                if not _selected(relative, include_paths):
                    continue
                if member.isdir():
                    continue
                if not member.isfile():
                    raise CorpusFetchError("corpus archive contains a selected non-regular file")
                if relative.suffix.lower() != ".md":
                    continue
                if relative in extracted_paths:
                    raise CorpusFetchError("corpus archive contains a duplicate selected path")
                extracted_paths.add(relative)

                extracted_bytes += member.size
                if extracted_bytes > max_extracted_bytes:
                    raise CorpusFetchError("extracted corpus exceeds the configured size limit")
                target = destination.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise CorpusFetchError("corpus archive path escapes the temporary directory")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise CorpusFetchError("corpus archive member has no readable content")
                _copy_exact(source, target, member.size)
    except (tarfile.TarError, OSError) as error:
        if isinstance(error, CorpusFetchError):
            raise
        raise CorpusFetchError("corpus archive could not be extracted") from error


def _safe_archive_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise CorpusFetchError("corpus archive contains an unsafe path")
    return path


def _selected(path: PurePosixPath, include_paths: tuple[PurePosixPath, ...]) -> bool:
    if any(part.startswith(".") for part in path.parts):
        return False
    return any(path == included or path.is_relative_to(included) for included in include_paths)


def _copy_exact(source: IO[bytes], target: Path, expected_size: int) -> None:
    written = 0
    with target.open("wb") as output:
        while block := source.read(_COPY_BLOCK_SIZE):
            written += len(block)
            if written > expected_size:
                raise CorpusFetchError("archive member expanded beyond its declared size")
            output.write(block)
    if written != expected_size:
        raise CorpusFetchError("archive member size does not match its declaration")


def _markdown_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (path for path in root.rglob("*.md") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
