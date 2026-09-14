"""BM25 retrieval specialized for exact SRE and Kubernetes identifiers."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from sre_rag.domain.documents import Chunk
from sre_rag.domain.retrieval import (
    RetrievalCandidate,
    RetrievalMethod,
    RetrievalQuery,
    RetrievalStage,
)

_TOKEN = re.compile(r"--[A-Za-z0-9_][A-Za-z0-9_.-]*|[A-Za-z0-9_]+(?:[./:-][A-Za-z0-9_]+)*")
_SEPARATOR = re.compile(r"[./:-]+")


class SRETokenizer:
    """Preserve operational identifiers while adding their component terms."""

    def tokenize(self, text: str) -> tuple[str, ...]:
        """Return normalized exact tokens followed by useful identifier parts."""

        output: list[str] = []
        for match in _TOKEN.finditer(text):
            exact = match.group().casefold()
            output.append(exact)

            without_flag_prefix = exact.removeprefix("--")
            parts = tuple(part for part in _SEPARATOR.split(without_flag_prefix) if part)
            if len(parts) > 1:
                output.extend(part for part in parts if part != exact)
        return tuple(output)


@dataclass(frozen=True, slots=True)
class BM25Config:
    """Validated BM25Okapi tuning and result-limit parameters."""

    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25
    default_limit: int = 10

    def __post_init__(self) -> None:
        if self.k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0 <= self.b <= 1:
            raise ValueError("b must be between zero and one")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if self.default_limit <= 0:
            raise ValueError("default limit must be positive")


class BM25Retriever:
    """Build an immutable sparse index and return typed ranked candidates."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        tokenizer: SRETokenizer | None = None,
        config: BM25Config | None = None,
    ) -> None:
        self._tokenizer = tokenizer or SRETokenizer()
        self._config = config or BM25Config()
        input_chunks = tuple(chunks)
        self._validate_unique_chunk_ids(input_chunks)

        indexed = tuple(
            (chunk, tokens)
            for chunk in input_chunks
            if (tokens := self._tokenizer.tokenize(chunk.text))
        )
        self._chunks = tuple(chunk for chunk, _ in indexed)
        self._token_sets = tuple(frozenset(tokens) for _, tokens in indexed)
        corpus = [list(tokens) for _, tokens in indexed]
        self._index = (
            BM25Okapi(
                corpus,
                k1=self._config.k1,
                b=self._config.b,
                epsilon=self._config.epsilon,
            )
            if corpus
            else None
        )

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        limit: int | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        """Rank matching chunks, excluding documents with no lexical evidence."""

        result_limit = self._config.default_limit if limit is None else limit
        if result_limit <= 0:
            raise ValueError("limit must be positive")
        if self._index is None:
            return ()

        query_tokens = self._tokenizer.tokenize(query.text)
        if not query_tokens:
            return ()

        scores = self._index.get_scores(list(query_tokens))
        query_token_set = frozenset(query_tokens)
        scored_chunks = [
            (float(score), chunk)
            for score, chunk, document_tokens in zip(
                scores,
                self._chunks,
                self._token_sets,
                strict=True,
            )
            if query_token_set & document_tokens
        ]
        scored_chunks.sort(key=lambda item: (-item[0], item[1].chunk_id))

        return tuple(
            RetrievalCandidate(
                chunk=chunk,
                stages=(
                    RetrievalStage(
                        method=RetrievalMethod.BM25,
                        score=score,
                        rank=rank,
                    ),
                ),
            )
            for rank, (score, chunk) in enumerate(scored_chunks[:result_limit], start=1)
        )

    @staticmethod
    def _validate_unique_chunk_ids(chunks: Sequence[Chunk]) -> None:
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("BM25 index requires unique chunk IDs")
