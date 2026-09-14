"""Markdown normalization and heading extraction."""

import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import yaml
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

_FRONT_MATTER_BOUNDARY = re.compile(r"^(---|\.\.\.)\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_SHORTCODE = re.compile(r"\{\{[<%]\s*(.*?)\s*[>%]\}\}")


@dataclass(frozen=True, slots=True)
class NormalizedMarkdown:
    """Normalized Markdown paired with its extracted title."""

    title: str
    content: str


@dataclass(frozen=True, slots=True)
class MarkdownSection:
    """Body text grouped under a Markdown heading hierarchy."""

    heading_path: tuple[str, ...]
    body: str


class MarkdownConverter(Protocol):
    """Adapter boundary for converting Markdown through a document parser."""

    def convert(self, content: str, *, name: str) -> str: ...


class DoclingMarkdownConverter:
    """Convert local Markdown through Docling with remote fetching disabled."""

    def __init__(self) -> None:
        self._converter = DocumentConverter(allowed_formats=[InputFormat.MD])

    def convert(self, content: str, *, name: str) -> str:
        result = self._converter.convert_string(content, format=InputFormat.MD, name=name)
        return result.document.export_to_markdown()


class MarkdownNormalizer:
    """Remove source metadata/templating before canonical Docling conversion."""

    def __init__(self, converter: MarkdownConverter) -> None:
        self._converter = converter

    def normalize(self, content: str, *, name: str) -> NormalizedMarkdown:
        metadata, body = split_front_matter(content)
        without_shortcodes = strip_hugo_shortcodes(body)
        normalized = self._converter.convert(without_shortcodes, name=name)
        title = _metadata_title(metadata) or first_heading(normalized) or _title_from_name(name)
        return NormalizedMarkdown(title=title, content=normalized)


def split_front_matter(content: str) -> tuple[Mapping[str, object], str]:
    """Extract optional YAML front matter without changing the remaining text."""

    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, content

    for index, line in enumerate(lines[1:], start=1):
        if _FRONT_MATTER_BOUNDARY.match(line):
            raw_metadata = "".join(lines[1:index])
            parsed = yaml.safe_load(raw_metadata) or {}
            if not isinstance(parsed, Mapping):
                raise ValueError("Markdown front matter must be a mapping")
            return cast(Mapping[str, object], parsed), "".join(lines[index + 1 :])
    raise ValueError("Markdown front matter is missing a closing boundary")


def strip_hugo_shortcodes(content: str) -> str:
    """Remove Hugo controls outside code fences while preserving display text."""

    output: list[str] = []
    fence_marker: str | None = None

    for line in content.splitlines(keepends=True):
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if fence_marker is None:
                fence_marker = marker
            elif marker == fence_marker:
                fence_marker = None

        output.append(line if fence_marker is not None else _SHORTCODE.sub(_shortcode_text, line))

    return "".join(output)


def _shortcode_text(match: re.Match[str]) -> str:
    expression = match.group(1).strip()
    if expression.startswith("/"):
        return ""

    try:
        tokens = shlex.split(expression)
    except ValueError:
        return ""

    for token in tokens[1:]:
        key, separator, value = token.partition("=")
        if separator and key == "text":
            return value
    return ""


def first_heading(content: str) -> str | None:
    """Return the first Markdown heading outside fenced code."""

    for section in split_markdown_sections(content):
        if section.heading_path:
            return section.heading_path[0]
    return None


def split_markdown_sections(content: str) -> tuple[MarkdownSection, ...]:
    """Group Markdown body text under heading paths without parsing fenced code."""

    sections: list[MarkdownSection] = []
    headings: list[str] = []
    body_lines: list[str] = []
    fence_marker: str | None = None

    def flush() -> None:
        body = "".join(body_lines).strip("\n")
        if body.strip():
            sections.append(MarkdownSection(tuple(headings), body))
        body_lines.clear()

    for line in content.splitlines(keepends=True):
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if fence_marker is None:
                fence_marker = marker
            elif marker == fence_marker:
                fence_marker = None
            body_lines.append(line)
            continue

        heading_match = _HEADING.match(line.rstrip("\r\n")) if fence_marker is None else None
        if heading_match:
            flush()
            level = len(heading_match.group(1))
            headings = headings[: level - 1]
            headings.append(heading_match.group(2).strip())
        else:
            body_lines.append(line)

    flush()
    return tuple(sections)


def _metadata_title(metadata: Mapping[str, object]) -> str | None:
    title = metadata.get("title")
    return title.strip() if isinstance(title, str) and title.strip() else None


def _title_from_name(name: str) -> str:
    stem = name.rsplit(".", maxsplit=1)[0]
    return stem.replace("-", "_").replace("_", " ").strip().title()
