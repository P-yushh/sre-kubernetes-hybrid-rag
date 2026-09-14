"""Heading-aware, token-bounded deterministic chunking."""

import re
from dataclasses import dataclass

from sre_rag.domain.documents import Chunk, SourceDocument
from sre_rag.ingestion.identifiers import chunk_id, sha256_text
from sre_rag.ingestion.markdown import MarkdownSection, split_markdown_sections
from sre_rag.ingestion.tokenizers import TokenSpanProvider

_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Token limits applied independently to every heading section."""

    max_tokens: int = 384
    overlap_tokens: int = 64

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("maximum token count must be positive")
        if self.overlap_tokens < 0 or self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap must be non-negative and smaller than the maximum")


@dataclass(frozen=True, slots=True)
class _MarkdownBlock:
    text: str
    fenced: bool


class HeadingAwareChunker:
    """Pack Markdown blocks under headings while respecting model token limits."""

    def __init__(self, tokenizer: TokenSpanProvider, config: ChunkingConfig | None = None) -> None:
        self._tokenizer = tokenizer
        self._config = config or ChunkingConfig()

    def chunk(self, document: SourceDocument) -> tuple[Chunk, ...]:
        """Create ordered, citable chunks from one normalized document."""

        chunks: list[Chunk] = []
        for section in split_markdown_sections(document.content):
            for text in self._chunk_section(section):
                ordinal = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id(document.document_id, section.heading_path, ordinal),
                        document_id=document.document_id,
                        text=text,
                        heading_path=section.heading_path,
                        ordinal=ordinal,
                        token_count=self._tokenizer.count(text),
                        content_sha256=sha256_text(text),
                        provenance=document.provenance,
                    )
                )
        return tuple(chunks)

    def _chunk_section(self, section: MarkdownSection) -> tuple[str, ...]:
        output: list[str] = []
        current: list[_MarkdownBlock] = []

        for block in _split_markdown_blocks(section.body):
            block_tokens = self._tokenizer.count(block.text)
            if block_tokens > self._config.max_tokens:
                if current:
                    output.append("".join(item.text for item in current).rstrip())
                    current = []
                output.extend(self._split_oversized_block(block.text))
                continue

            proposed = "".join(item.text for item in (*current, block))
            if current and self._tokenizer.count(proposed) > self._config.max_tokens:
                previous = "".join(item.text for item in current).rstrip()
                output.append(previous)
                current = self._overlap_blocks(current)
                proposed = "".join(item.text for item in (*current, block))
                if self._tokenizer.count(proposed) > self._config.max_tokens:
                    current = []

            current.append(block)

        if current:
            output.append("".join(item.text for item in current).rstrip())
        return tuple(text for text in output if text.strip())

    def _overlap_blocks(self, blocks: list[_MarkdownBlock]) -> list[_MarkdownBlock]:
        if self._config.overlap_tokens == 0:
            return []

        selected: list[_MarkdownBlock] = []
        for block in reversed(blocks):
            candidate = [block, *selected]
            if self._tokenizer.count("".join(item.text for item in candidate)) > (
                self._config.overlap_tokens
            ):
                break
            selected = candidate

        if selected or not blocks or blocks[-1].fenced:
            return selected

        suffix = self._token_suffix(blocks[-1].text, self._config.overlap_tokens)
        return [_MarkdownBlock(suffix, fenced=False)] if suffix else []

    def _split_oversized_block(self, text: str) -> tuple[str, ...]:
        spans = self._tokenizer.token_spans(text)
        if not spans:
            return ()

        windows: list[str] = []
        start = 0
        while start < len(spans):
            end = min(start + self._config.max_tokens, len(spans))
            char_start = 0 if start == 0 else spans[start].start
            char_end = len(text) if end == len(spans) else spans[end].start
            windows.append(text[char_start:char_end].rstrip())
            if end == len(spans):
                break
            start = end - self._config.overlap_tokens
        return tuple(windows)

    def _token_suffix(self, text: str, size: int) -> str:
        spans = self._tokenizer.token_spans(text)
        if not spans:
            return ""
        start = spans[max(0, len(spans) - size)].start
        return text[start:]


def _split_markdown_blocks(content: str) -> tuple[_MarkdownBlock, ...]:
    """Split paragraphs and fenced code while preserving original characters."""

    blocks: list[_MarkdownBlock] = []
    current: list[str] = []
    fence_marker: str | None = None

    def flush(*, fenced: bool) -> None:
        if current:
            blocks.append(_MarkdownBlock("".join(current), fenced=fenced))
            current.clear()

    for line in content.splitlines(keepends=True):
        fence_match = _FENCE.match(line)
        if fence_marker is not None:
            current.append(line)
            if fence_match and fence_match.group(1)[0] == fence_marker:
                flush(fenced=True)
                fence_marker = None
            continue

        if fence_match:
            flush(fenced=False)
            fence_marker = fence_match.group(1)[0]
            current.append(line)
        else:
            current.append(line)
            if not line.strip():
                flush(fenced=False)

    flush(fenced=fence_marker is not None)
    return tuple(blocks)
