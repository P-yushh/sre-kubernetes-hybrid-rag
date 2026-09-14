"""Rank-fusion interfaces and deterministic Reciprocal Rank Fusion."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from sre_rag.domain.documents import Chunk
from sre_rag.domain.retrieval import (
    RetrievalCandidate,
    RetrievalMethod,
    RetrievalStage,
)

Rankings = Mapping[RetrievalMethod, Sequence[RetrievalCandidate]]
_SUPPORTED_SOURCES = frozenset({RetrievalMethod.BM25, RetrievalMethod.DENSE})
_STAGE_ORDER = {
    RetrievalMethod.BM25: 0,
    RetrievalMethod.DENSE: 1,
    RetrievalMethod.RRF: 2,
    RetrievalMethod.RERANKER: 3,
}


class RankFusion(Protocol):
    """Combine independently scored rankings without comparing raw scores."""

    def fuse(
        self,
        rankings: Rankings,
        *,
        limit: int | None = None,
    ) -> tuple[RetrievalCandidate, ...]: ...


@dataclass(frozen=True, slots=True)
class RRFConfig:
    """Parameters for Reciprocal Rank Fusion."""

    rank_constant: int = 60
    default_limit: int = 20

    def __post_init__(self) -> None:
        if self.rank_constant <= 0:
            raise ValueError("RRF rank constant must be positive")
        if self.default_limit <= 0:
            raise ValueError("RRF default limit must be positive")


@dataclass(slots=True)
class _FusedCandidate:
    chunk: Chunk
    stages: list[RetrievalStage]
    score: float = 0.0


class ReciprocalRankFusion:
    """Fuse sparse and dense ranks using equal reciprocal-rank contributions."""

    def __init__(self, config: RRFConfig | None = None) -> None:
        self._config = config or RRFConfig()

    def fuse(
        self,
        rankings: Rankings,
        *,
        limit: int | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        """Return candidates ordered by summed ``1 / (k + rank)`` scores."""

        result_limit = self._config.default_limit if limit is None else limit
        if result_limit <= 0:
            raise ValueError("fusion limit must be positive")
        self._validate_source_methods(rankings)

        fused: dict[str, _FusedCandidate] = {}
        for source_method, candidates in rankings.items():
            self._validate_ranking(source_method, candidates)
            for candidate in candidates:
                stage = candidate.stages[0]
                current = fused.get(candidate.chunk.chunk_id)
                if current is None:
                    current = _FusedCandidate(chunk=candidate.chunk, stages=[])
                    fused[candidate.chunk.chunk_id] = current
                elif current.chunk != candidate.chunk:
                    raise ValueError(
                        f"chunk ID {candidate.chunk.chunk_id!r} has inconsistent payloads"
                    )

                current.stages.append(stage)
                current.score += 1.0 / (self._config.rank_constant + stage.rank)

        ordered = sorted(fused.values(), key=lambda item: (-item.score, item.chunk.chunk_id))
        return tuple(
            RetrievalCandidate(
                chunk=item.chunk,
                stages=(
                    *sorted(item.stages, key=lambda stage: _STAGE_ORDER[stage.method]),
                    RetrievalStage(
                        method=RetrievalMethod.RRF,
                        score=item.score,
                        rank=rank,
                    ),
                ),
            )
            for rank, item in enumerate(ordered[:result_limit], start=1)
        )

    @staticmethod
    def _validate_source_methods(rankings: Rankings) -> None:
        unsupported = set(rankings) - _SUPPORTED_SOURCES
        if unsupported:
            names = ", ".join(sorted(method.value for method in unsupported))
            raise ValueError(f"unsupported RRF source methods: {names}")

    @staticmethod
    def _validate_ranking(
        source_method: RetrievalMethod,
        candidates: Sequence[RetrievalCandidate],
    ) -> None:
        chunk_ids: set[str] = set()
        ranks: set[int] = set()
        for candidate in candidates:
            if len(candidate.stages) != 1 or candidate.stages[0].method is not source_method:
                raise ValueError(
                    f"every {source_method.value} candidate must contain only its source stage"
                )
            stage = candidate.stages[0]
            if candidate.chunk.chunk_id in chunk_ids:
                raise ValueError(
                    f"duplicate chunk ID {candidate.chunk.chunk_id!r} "
                    f"in {source_method.value} ranking"
                )
            if stage.rank in ranks:
                raise ValueError(f"duplicate rank {stage.rank} in {source_method.value} ranking")
            chunk_ids.add(candidate.chunk.chunk_id)
            ranks.add(stage.rank)
