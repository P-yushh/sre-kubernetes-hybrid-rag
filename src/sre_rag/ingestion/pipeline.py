"""Composition of corpus loading and deterministic chunking."""

from dataclasses import dataclass

from sre_rag.domain.documents import Chunk, SourceDocument
from sre_rag.ingestion.chunking import HeadingAwareChunker
from sre_rag.ingestion.loaders import CorpusConfig, MarkdownCorpusLoader


@dataclass(frozen=True, slots=True)
class IngestionBatch:
    """Documents and chunks produced from one versioned corpus snapshot."""

    documents: tuple[SourceDocument, ...]
    chunks: tuple[Chunk, ...]


class IngestionPipeline:
    """Load and chunk a corpus without coupling either implementation."""

    def __init__(self, loader: MarkdownCorpusLoader, chunker: HeadingAwareChunker) -> None:
        self._loader = loader
        self._chunker = chunker

    def run(self, config: CorpusConfig) -> IngestionBatch:
        """Return a deterministic batch ready for retrieval indexing."""

        documents = self._loader.load(config)
        chunks = tuple(chunk for document in documents for chunk in self._chunker.chunk(document))
        return IngestionBatch(documents=documents, chunks=chunks)
