"""Safe, deterministic loading of versioned Markdown corpora."""

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from sre_rag.domain.documents import Corpus, SourceDocument, SourceProvenance
from sre_rag.ingestion.identifiers import document_id, sha256_text
from sre_rag.ingestion.markdown import MarkdownNormalizer


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    """Versioned local corpus boundaries and citation metadata."""

    root: Path
    corpus: Corpus
    repository: str
    revision: str
    license: str
    include_paths: tuple[PurePosixPath, ...]
    max_file_size_bytes: int = 2 * 1024 * 1024
    ignored_directories: frozenset[str] = field(
        default_factory=lambda: frozenset({"node_modules", "public", "resources"})
    )

    def __post_init__(self) -> None:
        if self.max_file_size_bytes <= 0:
            raise ValueError("maximum file size must be positive")
        if not self.include_paths:
            raise ValueError("at least one include path is required")


class MarkdownCorpusLoader:
    """Load allowed Markdown paths into reproducible source documents."""

    def __init__(self, normalizer: MarkdownNormalizer) -> None:
        self._normalizer = normalizer

    def discover(self, config: CorpusConfig) -> tuple[Path, ...]:
        """Return normalized Markdown paths in deterministic order."""

        root = config.root.resolve(strict=True)
        discovered: set[Path] = set()

        for include_path in config.include_paths:
            self._validate_include_path(include_path)
            target = (root / include_path).resolve(strict=True)
            self._ensure_within_root(target, root)

            candidates = (target,) if target.is_file() else target.rglob("*.md")
            for candidate in candidates:
                if candidate.suffix.lower() != ".md" or self._is_ignored(candidate, root, config):
                    continue
                resolved = candidate.resolve(strict=True)
                self._ensure_within_root(resolved, root)
                discovered.add(resolved)

        return tuple(sorted(discovered, key=lambda path: path.relative_to(root).as_posix()))

    def load(self, config: CorpusConfig) -> tuple[SourceDocument, ...]:
        """Normalize discovered files and attach immutable provenance."""

        root = config.root.resolve(strict=True)
        documents: list[SourceDocument] = []

        for path in self.discover(config):
            if path.stat().st_size > config.max_file_size_bytes:
                raise ValueError(f"Markdown file exceeds size limit: {path}")

            relative_path = PurePosixPath(path.relative_to(root).as_posix())
            raw_content = path.read_text(encoding="utf-8")
            normalized = self._normalizer.normalize(raw_content, name=path.name)
            source_url = self._source_url(config.repository, config.revision, relative_path)
            provenance = SourceProvenance(
                corpus=config.corpus,
                repository=config.repository,
                revision=config.revision,
                relative_path=relative_path.as_posix(),
                source_url=source_url,
                license=config.license,
            )
            documents.append(
                SourceDocument(
                    document_id=document_id(config.corpus, relative_path),
                    title=normalized.title,
                    content=normalized.content,
                    content_sha256=sha256_text(normalized.content),
                    provenance=provenance,
                )
            )

        return tuple(documents)

    @staticmethod
    def _validate_include_path(path: PurePosixPath) -> None:
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("include paths must stay inside the corpus root")

    @staticmethod
    def _ensure_within_root(path: Path, root: Path) -> None:
        if not path.is_relative_to(root):
            raise ValueError(f"corpus path escapes configured root: {path}")

    @staticmethod
    def _is_ignored(path: Path, root: Path, config: CorpusConfig) -> bool:
        parts = path.relative_to(root).parts
        return any(part.startswith(".") or part in config.ignored_directories for part in parts)

    @staticmethod
    def _source_url(repository: str, revision: str, relative_path: PurePosixPath) -> str:
        encoded_path = quote(relative_path.as_posix(), safe="/")
        return f"https://github.com/{repository}/blob/{revision}/{encoded_path}"
