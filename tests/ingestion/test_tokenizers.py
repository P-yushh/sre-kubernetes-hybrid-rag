"""Tests for the Hugging Face tokenizer adapter."""

from typing import Any, cast

import pytest
from transformers import PreTrainedTokenizerBase

from sre_rag.ingestion.tokenizers import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL_REVISION,
    HuggingFaceTokenizer,
)


class FakeFastTokenizer:
    is_fast = True

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[tuple[int, int]]]:
        del kwargs
        first_end = text.index(" ")
        return {"offset_mapping": [(0, first_end), (first_end + 1, len(text))]}


class FakeSlowTokenizer(FakeFastTokenizer):
    is_fast = False


def test_hugging_face_adapter_exposes_character_offsets() -> None:
    backend = cast(PreTrainedTokenizerBase, FakeFastTokenizer())
    tokenizer = HuggingFaceTokenizer(backend)

    spans = tokenizer.token_spans("exit 137")

    assert [(span.start, span.end) for span in spans] == [(0, 4), (5, 8)]
    assert tokenizer.count("exit 137") == 2


def test_hugging_face_adapter_requires_fast_tokenizer() -> None:
    backend = cast(PreTrainedTokenizerBase, FakeSlowTokenizer())

    with pytest.raises(ValueError, match="fast Hugging Face"):
        HuggingFaceTokenizer(backend)


def test_pretrained_tokenizer_uses_pinned_model_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_from_pretrained(model_name: str, **kwargs: object) -> PreTrainedTokenizerBase:
        captured.update(model_name=model_name, **kwargs)
        return cast(PreTrainedTokenizerBase, FakeFastTokenizer())

    monkeypatch.setattr(
        "sre_rag.ingestion.tokenizers.AutoTokenizer.from_pretrained", fake_from_pretrained
    )

    tokenizer = HuggingFaceTokenizer.from_pretrained()

    assert isinstance(tokenizer, HuggingFaceTokenizer)
    assert captured["model_name"] == DEFAULT_EMBEDDING_MODEL
    assert captured["revision"] == DEFAULT_EMBEDDING_MODEL_REVISION
    assert captured["trust_remote_code"] is False
