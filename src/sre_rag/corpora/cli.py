"""Command-line verification of pinned remote corpus snapshots."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from sre_rag.corpora.fetch import CorpusFetcher, HttpxArchiveDownloader
from sre_rag.corpora.manifest import load_manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch, validate, report, and remove every configured corpus snapshot."""

    parser = argparse.ArgumentParser(description="Verify pinned RAG corpus snapshots")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("corpora/manifest.yaml"),
        help="Versioned corpus manifest",
    )
    arguments = parser.parse_args(argv)
    manifest = load_manifest(arguments.manifest)
    downloader = HttpxArchiveDownloader()
    fetcher = CorpusFetcher(downloader)
    try:
        for source in manifest.sources:
            with fetcher.materialize(source) as corpus:
                print(
                    f"{source.name}: {len(corpus.markdown_files)} Markdown files "
                    f"verified at {source.revision}"
                )
    finally:
        downloader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
