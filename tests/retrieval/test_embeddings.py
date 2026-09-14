"""Tests for BGE embedding behavior without downloading model weights."""

from typing import Any

import numpy as np
import pytest
import torch

from sre_rag.retrieval.embeddings import (
    BGE_QUERY_INSTRUCTION,
    BGEConfig,
    BGEEmbedder,
    select_torch_device,
)


class FakeSentenceEncoder:
    def __init__(self, output: object, *, dimension: int | None = 3) -> None:
        self.output = output
        self.dimension = dimension
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def get_sentence_embedding_dimension(self) -> int | None:
        return self.dimension

    def encode(self, inputs: list[str], **kwargs: Any) -> object:
        self.calls.append((inputs, kwargs))
        return self.output


def test_documents_are_normalized_without_query_instruction() -> None:
    encoder = FakeSentenceEncoder(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    embedder = BGEEmbedder(encoder, BGEConfig(batch_size=2))

    vectors = embedder.embed_documents(["CrashLoopBackOff", "readiness probe"])

    assert vectors == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    inputs, options = encoder.calls[0]
    assert inputs == ["CrashLoopBackOff", "readiness probe"]
    assert options == {
        "batch_size": 2,
        "show_progress_bar": False,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }


def test_query_uses_bge_retrieval_instruction() -> None:
    encoder = FakeSentenceEncoder(np.array([[0.0, 0.0, 1.0]]))
    embedder = BGEEmbedder(encoder)

    vector = embedder.embed_query("Why was the pod OOMKilled?")

    assert vector == (0.0, 0.0, 1.0)
    assert encoder.calls[0][0] == [f"{BGE_QUERY_INSTRUCTION}Why was the pod OOMKilled?"]


def test_empty_documents_do_not_invoke_model() -> None:
    encoder = FakeSentenceEncoder(np.empty((0, 3)))

    assert BGEEmbedder(encoder).embed_documents([]) == ()
    assert encoder.calls == []


@pytest.mark.parametrize(
    "output",
    [np.array([[1.0, 0.0]]), np.array([[1.0, float("nan"), 0.0]])],
)
def test_invalid_embedding_output_is_rejected(output: object) -> None:
    with pytest.raises(ValueError, match="embedding output|non-finite"):
        BGEEmbedder(FakeSentenceEncoder(output)).embed_query("OOMKilled")


@pytest.mark.parametrize("dimension", [None, 0, -1])
def test_invalid_model_dimension_is_rejected(dimension: int | None) -> None:
    with pytest.raises(ValueError, match="positive vector dimension"):
        BGEEmbedder(FakeSentenceEncoder(np.empty((0, 0)), dimension=dimension))


def test_blank_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        BGEEmbedder(FakeSentenceEncoder(np.empty((0, 3)))).embed_query(" \n")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_name": " "},
        {"revision": " "},
        {"query_instruction": " "},
        {"batch_size": 0},
        {"device": "cuda"},
    ],
)
def test_invalid_embedding_config_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        BGEConfig(**kwargs)  # type: ignore[arg-type]


def test_device_selection_prefers_mps_and_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_torch_device() == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_torch_device() == "cpu"
    assert select_torch_device("mps") == "mps"
