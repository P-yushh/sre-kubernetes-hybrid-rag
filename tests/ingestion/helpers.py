"""Deterministic test doubles for external parsing and tokenization."""

import re

from sre_rag.ingestion.tokenizers import TokenSpan


class PassThroughConverter:
    """Return preprocessed Markdown without external normalization."""

    def convert(self, content: str, *, name: str) -> str:
        del name
        return content


class RegexTokenizer:
    """Treat each non-whitespace run as one token for predictable tests."""

    def token_spans(self, text: str) -> tuple[TokenSpan, ...]:
        return tuple(TokenSpan(match.start(), match.end()) for match in re.finditer(r"\S+", text))

    def count(self, text: str) -> int:
        return len(self.token_spans(text))
