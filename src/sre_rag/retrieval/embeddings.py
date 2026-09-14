"""Dense embedding interfaces and the BGE Sentence Transformers adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray

from sre_rag.ingestion.tokenizers import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL_REVISION,
)

Vector = tuple[float, ...]
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    """Create compatible normalized vectors for passages and queries."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]: ...

    def embed_query(self, text: str) -> Vector: ...


class SentenceEncoder(Protocol):
    """Narrow surface used from Sentence Transformers for easy substitution."""

    def get_sentence_embedding_dimension(self) -> int | None: ...

    def encode(
        self,
        inputs: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class BGEConfig:
    """Configuration that keeps embedding behavior reproducible."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    revision: str = DEFAULT_EMBEDDING_MODEL_REVISION
    query_instruction: str = BGE_QUERY_INSTRUCTION
    batch_size: int = 8
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("embedding model name must not be blank")
        if not self.revision.strip():
            raise ValueError("embedding model revision must not be blank")
        if not self.query_instruction.strip():
            raise ValueError("query instruction must not be blank")
        if self.batch_size <= 0:
            raise ValueError("embedding batch size must be positive")
        if self.device not in {"auto", "mps", "cpu"}:
            raise ValueError("embedding device must be auto, mps, or cpu")


def select_torch_device(configured_device: str = "auto") -> str:
    """Select Apple Metal when available, otherwise use the CPU."""

    if configured_device != "auto":
        return configured_device
    return "mps" if torch.backends.mps.is_available() else "cpu"


class BGEEmbedder:
    """Generate normalized BGE vectors with asymmetric query prompting."""

    def __init__(self, encoder: SentenceEncoder, config: BGEConfig | None = None) -> None:
        self._encoder = encoder
        self._config = config or BGEConfig()
        dimension = encoder.get_sentence_embedding_dimension()
        if dimension is None or dimension <= 0:
            raise ValueError("embedding model must report a positive vector dimension")
        self._dimension = dimension

    @classmethod
    def from_pretrained(cls, config: BGEConfig | None = None) -> BGEEmbedder:
        """Load the pinned Sentence Transformers model only when requested."""

        from sentence_transformers import SentenceTransformer

        resolved = config or BGEConfig()
        encoder = SentenceTransformer(
            resolved.model_name,
            revision=resolved.revision,
            device=select_torch_device(resolved.device),
            trust_remote_code=False,
            backend="torch",
        )
        return cls(cast(SentenceEncoder, encoder), resolved)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        """Embed passages without the retrieval query instruction."""

        if not texts:
            return ()
        return self._encode(list(texts))

    def embed_query(self, text: str) -> Vector:
        """Embed one query with the instruction expected by BGE v1.5."""

        if not text.strip():
            raise ValueError("embedding query must not be blank")
        vectors = self._encode([f"{self._config.query_instruction}{text}"])
        return vectors[0]

    def _encode(self, inputs: list[str]) -> tuple[Vector, ...]:
        encoded = self._encoder.encode(
            inputs,
            batch_size=self._config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        array = np.asarray(encoded, dtype=np.float32)
        self._validate_array(array, expected_rows=len(inputs))
        return tuple(tuple(float(value) for value in row) for row in array)

    def _validate_array(self, array: NDArray[np.float32], *, expected_rows: int) -> None:
        if array.shape != (expected_rows, self._dimension):
            raise ValueError(
                "embedding output shape mismatch: "
                f"expected {(expected_rows, self._dimension)}, received {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("embedding output contains a non-finite value")
