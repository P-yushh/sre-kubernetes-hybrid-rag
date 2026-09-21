"""Independent validation and resolution of model-produced citations."""

import re
from collections.abc import Sequence

from sre_rag.domain.answers import Citation, GroundedAnswer
from sre_rag.domain.generation import GeneratedAnswerDraft
from sre_rag.domain.retrieval import RetrievalCandidate

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class CitationValidationError(ValueError):
    """A generated draft does not map exactly to the supplied evidence."""


class CitationValidator:
    """Resolve inline context numbers to immutable source metadata."""

    def validate(
        self,
        draft: GeneratedAnswerDraft,
        contexts: Sequence[RetrievalCandidate],
    ) -> GroundedAnswer:
        """Return a public answer only when every citation is valid and resolvable."""

        if not draft.answerable:
            raise CitationValidationError("an unanswerable draft cannot become an answer")

        inline_numbers = {int(value) for value in _CITATION_PATTERN.findall(draft.answer)}
        declared_numbers = set(draft.citation_numbers)
        if inline_numbers != declared_numbers:
            raise CitationValidationError(
                "inline citation markers do not match declared citation numbers"
            )

        context_count = len(contexts)
        if any(number > context_count for number in declared_numbers):
            raise CitationValidationError("citation number is outside the supplied context range")

        citations = tuple(
            _citation_from_candidate(number, contexts[number - 1])
            for number in sorted(declared_numbers)
        )
        return GroundedAnswer(answer=draft.answer, citations=citations)


def _citation_from_candidate(number: int, candidate: RetrievalCandidate) -> Citation:
    chunk = candidate.chunk
    title = " > ".join(chunk.heading_path)
    if not title:
        title = chunk.provenance.relative_path.rsplit("/", maxsplit=1)[-1]
    return Citation(
        number=number,
        chunk_id=chunk.chunk_id,
        source_title=title,
        source_path=chunk.provenance.relative_path,
        source_url=chunk.provenance.source_url,
    )
