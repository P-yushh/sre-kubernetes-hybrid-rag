"""Deterministic identifiers and content fingerprints."""

import hashlib
import re
import unicodedata
from pathlib import PurePosixPath

from sre_rag.domain.documents import Corpus

_NON_SLUG_CHARACTER = re.compile(r"[^a-z0-9]+")


def sha256_text(value: str) -> str:
    """Return the lowercase SHA-256 digest of UTF-8 text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def document_id(corpus: Corpus, relative_path: PurePosixPath) -> str:
    """Build a readable identity from a corpus and normalized source path."""

    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("document paths must be relative and cannot contain '..'")
    return f"{corpus.value}:{relative_path.as_posix()}"


def heading_slug(heading_path: tuple[str, ...]) -> str:
    """Create a stable ASCII slug from a complete heading hierarchy."""

    components = tuple(filter(None, (_slug_component(heading) for heading in heading_path)))
    return "--".join(components) or "root"


def _slug_component(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return _NON_SLUG_CHARACTER.sub("-", normalized.lower()).strip("-")


def chunk_id(parent_document_id: str, heading_path: tuple[str, ...], ordinal: int) -> str:
    """Build a deterministic, human-readable chunk identity."""

    if ordinal < 0:
        raise ValueError("chunk ordinal cannot be negative")
    return f"{parent_document_id}#{heading_slug(heading_path)}@{ordinal}"
