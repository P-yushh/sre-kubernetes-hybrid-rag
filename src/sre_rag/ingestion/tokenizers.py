"""Token offset adapters used by deterministic chunking."""

from dataclasses import dataclass
from typing import Protocol, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_EMBEDDING_MODEL_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """Half-open character offsets for one model token."""

    start: int
    end: int


class TokenSpanProvider(Protocol):
    """Minimal tokenizer behavior required by the chunker."""

    def token_spans(self, text: str) -> tuple[TokenSpan, ...]: ...

    def count(self, text: str) -> int: ...


class HuggingFaceTokenizer:
    """Expose fast-tokenizer offsets without leaking Transformers into the chunker."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        if not tokenizer.is_fast:
            raise ValueError("a fast Hugging Face tokenizer is required for offset mapping")
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        revision: str = DEFAULT_EMBEDDING_MODEL_REVISION,
    ) -> "HuggingFaceTokenizer":
        """Load the tokenizer associated with the configured embedding model."""

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            use_fast=True,
            trust_remote_code=False,
        )
        return cls(tokenizer)

    def token_spans(self, text: str) -> tuple[TokenSpan, ...]:
        """Return character offsets for non-special tokens in the input."""

        encoding = self._tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_offsets_mapping=True,
        )
        offsets = cast(list[tuple[int, int]], encoding["offset_mapping"])
        return tuple(TokenSpan(start, end) for start, end in offsets if end > start)

    def count(self, text: str) -> int:
        """Count tokens exactly as the embedding tokenizer does."""

        return len(self.token_spans(text))
