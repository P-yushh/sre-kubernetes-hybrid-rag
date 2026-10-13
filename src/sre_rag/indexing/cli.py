"""Production command for building local hybrid retrieval indexes."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from qdrant_client import QdrantClient

from sre_rag.config import get_settings
from sre_rag.corpora.fetch import CorpusFetcher, HttpxArchiveDownloader
from sre_rag.corpora.manifest import load_manifest
from sre_rag.indexing.artifacts import sha256_file
from sre_rag.indexing.builder import CorpusIndexBuilder
from sre_rag.ingestion.chunking import HeadingAwareChunker
from sre_rag.ingestion.loaders import MarkdownCorpusLoader
from sre_rag.ingestion.markdown import DoclingMarkdownConverter, MarkdownNormalizer
from sre_rag.ingestion.pipeline import IngestionPipeline
from sre_rag.ingestion.tokenizers import HuggingFaceTokenizer
from sre_rag.retrieval.dense import DenseRetriever
from sre_rag.retrieval.embeddings import BGEEmbedder
from sre_rag.retrieval.qdrant import QdrantVectorStore


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch pinned corpora and build deterministic sparse and dense indexes."""

    arguments = _parser().parse_args(argv)
    settings = get_settings()
    manifest = load_manifest(arguments.manifest)
    dense_collection = f"{settings.qdrant_collection_name}_{sha256_file(arguments.manifest)[:12]}"
    downloader = HttpxArchiveDownloader()
    qdrant = QdrantClient(url=settings.qdrant_url)
    try:
        tokenizer = HuggingFaceTokenizer.from_pretrained()
        embedder = BGEEmbedder.from_pretrained()
        dense = DenseRetriever(
            embedder,
            QdrantVectorStore(qdrant, collection_name=dense_collection),
        )
        builder = CorpusIndexBuilder(
            CorpusFetcher(downloader),
            IngestionPipeline(
                MarkdownCorpusLoader(MarkdownNormalizer(DoclingMarkdownConverter())),
                HeadingAwareChunker(tokenizer),
            ),
            dense,
        )
        report = builder.build(
            manifest,
            manifest_path=arguments.manifest,
            chunks_path=arguments.chunks,
            metadata_path=arguments.metadata,
            dense_collection=dense_collection,
        )
    finally:
        downloader.close()
        qdrant.close()

    print(
        f"Indexed {report.artifact.chunk_count} chunks from "
        f"{len(report.artifact.sources)} pinned corpora."
    )
    print(f"Sparse artifact: {report.chunks_path}")
    print(f"Index metadata: {report.metadata_path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build SRE RAG hybrid indexes")
    parser.add_argument("--manifest", type=Path, default=Path("corpora/manifest.yaml"))
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/index-manifest.json"),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
