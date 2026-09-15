"""Cross-encoder reranking with pinned, interchangeable model backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np

from sre_rag.domain.retrieval import (
    RetrievalCandidate,
    RetrievalMethod,
    RetrievalQuery,
    RetrievalStage,
)
from sre_rag.retrieval.embeddings import select_torch_device

MINILM_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
MINILM_RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
BGE_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


class Reranker(Protocol):
    """Reorder a bounded candidate set using joint query-document scoring."""

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalCandidate],
        *,
        limit: int | None = None,
    ) -> tuple[RetrievalCandidate, ...]: ...


class CrossEncoderBackend(Protocol):
    """Narrow Sentence Transformers prediction surface used by the adapter."""

    def predict(
        self,
        inputs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        apply_softmax: bool,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class CrossEncoderConfig:
    """Reproducible model and inference limits for cross-encoder scoring."""

    model_name: str = MINILM_RERANKER_MODEL
    revision: str = MINILM_RERANKER_REVISION
    batch_size: int = 8
    max_length: int = 512
    max_candidates: int = 50
    default_limit: int = 5
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("reranker model name must not be blank")
        if not self.revision.strip():
            raise ValueError("reranker model revision must not be blank")
        if self.batch_size <= 0:
            raise ValueError("reranker batch size must be positive")
        if self.max_length <= 0:
            raise ValueError("reranker maximum length must be positive")
        if self.max_candidates <= 0:
            raise ValueError("reranker candidate limit must be positive")
        if self.default_limit <= 0 or self.default_limit > self.max_candidates:
            raise ValueError("reranker result limit must be positive and within candidate limit")
        if self.device not in {"auto", "mps", "cpu"}:
            raise ValueError("reranker device must be auto, mps, or cpu")

    @classmethod
    def minilm(
        cls,
        *,
        batch_size: int = 8,
        max_length: int = 512,
        max_candidates: int = 50,
        default_limit: int = 5,
        device: str = "auto",
    ) -> CrossEncoderConfig:
        """Create the fast English reranker configuration."""

        return cls(
            model_name=MINILM_RERANKER_MODEL,
            revision=MINILM_RERANKER_REVISION,
            batch_size=batch_size,
            max_length=max_length,
            max_candidates=max_candidates,
            default_limit=default_limit,
            device=device,
        )

    @classmethod
    def bge(
        cls,
        *,
        batch_size: int = 4,
        max_length: int = 512,
        max_candidates: int = 50,
        default_limit: int = 5,
        device: str = "auto",
    ) -> CrossEncoderConfig:
        """Create the higher-quality multilingual reranker configuration."""

        return cls(
            model_name=BGE_RERANKER_MODEL,
            revision=BGE_RERANKER_REVISION,
            batch_size=batch_size,
            max_length=max_length,
            max_candidates=max_candidates,
            default_limit=default_limit,
            device=device,
        )


class CrossEncoderReranker:
    """Append neural relevance scores to a validated fused ranking."""

    def __init__(
        self,
        backend: CrossEncoderBackend,
        config: CrossEncoderConfig | None = None,
    ) -> None:
        self._backend = backend
        self._config = config or CrossEncoderConfig.minilm()

    @classmethod
    def from_pretrained(
        cls,
        config: CrossEncoderConfig | None = None,
    ) -> CrossEncoderReranker:
        """Load the configured PyTorch model only when runtime composition requests it."""

        import torch
        from sentence_transformers import CrossEncoder

        resolved = config or CrossEncoderConfig.minilm()
        backend = CrossEncoder(
            resolved.model_name,
            revision=resolved.revision,
            device=select_torch_device(resolved.device),
            trust_remote_code=False,
            backend="torch",
            max_length=resolved.max_length,
            activation_fn=torch.nn.Identity(),
        )
        return cls(cast(CrossEncoderBackend, backend), resolved)

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalCandidate],
        *,
        limit: int | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        """Score all fused candidates jointly with the query, then return top results."""

        result_limit = self._config.default_limit if limit is None else limit
        if result_limit <= 0 or result_limit > self._config.max_candidates:
            raise ValueError("reranker limit must be positive and within candidate limit")
        if len(candidates) > self._config.max_candidates:
            raise ValueError(
                f"reranker received {len(candidates)} candidates; "
                f"maximum is {self._config.max_candidates}"
            )
        if not candidates:
            return ()
        self._validate_candidates(candidates)

        pairs = [(query.text, candidate.chunk.text) for candidate in candidates]
        raw_scores = self._backend.predict(
            pairs,
            batch_size=self._config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            apply_softmax=False,
        )
        scores = self._validate_scores(raw_scores, expected_count=len(candidates))
        scored = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-item[1], item[0].chunk.chunk_id),
        )

        return tuple(
            RetrievalCandidate(
                chunk=candidate.chunk,
                stages=(
                    *candidate.stages,
                    RetrievalStage(
                        method=RetrievalMethod.RERANKER,
                        score=score,
                        rank=rank,
                    ),
                ),
            )
            for rank, (candidate, score) in enumerate(scored[:result_limit], start=1)
        )

    @staticmethod
    def _validate_candidates(candidates: Sequence[RetrievalCandidate]) -> None:
        chunk_ids: set[str] = set()
        rrf_ranks: set[int] = set()
        for candidate in candidates:
            methods = {stage.method for stage in candidate.stages}
            if RetrievalMethod.RRF not in methods:
                raise ValueError("reranker candidates must contain an RRF stage")
            if RetrievalMethod.RERANKER in methods:
                raise ValueError("candidate has already been reranked")
            if candidate.chunk.chunk_id in chunk_ids:
                raise ValueError(f"duplicate reranker chunk ID {candidate.chunk.chunk_id!r}")
            rrf_rank = next(
                stage.rank for stage in candidate.stages if stage.method is RetrievalMethod.RRF
            )
            if rrf_rank in rrf_ranks:
                raise ValueError(f"duplicate RRF rank {rrf_rank} in reranker candidates")
            chunk_ids.add(candidate.chunk.chunk_id)
            rrf_ranks.add(rrf_rank)

    @staticmethod
    def _validate_scores(raw_scores: object, *, expected_count: int) -> tuple[float, ...]:
        scores = np.asarray(raw_scores, dtype=np.float32)
        if scores.shape == (expected_count, 1):
            scores = scores.reshape(expected_count)
        if scores.shape != (expected_count,):
            raise ValueError(
                "reranker output shape mismatch: "
                f"expected {(expected_count,)}, received {scores.shape}"
            )
        if not np.isfinite(scores).all():
            raise ValueError("reranker output contains a non-finite value")
        return tuple(float(score) for score in scores)
