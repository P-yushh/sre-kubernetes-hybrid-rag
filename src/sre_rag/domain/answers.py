"""Grounded-answer, citation, and deterministic-refusal contracts."""

import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, HttpUrl, model_validator

from sre_rag.domain.base import DomainModel, NonEmptyText

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class Citation(DomainModel):
    """A numbered inline citation mapped to an indexed source chunk."""

    number: int = Field(gt=0)
    chunk_id: NonEmptyText
    source_title: NonEmptyText
    source_path: NonEmptyText
    source_url: HttpUrl


class GroundedAnswer(DomainModel):
    """An answer whose inline markers map exactly to structured citations."""

    outcome: Literal["answer"] = "answer"
    answer: NonEmptyText
    citations: tuple[Citation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def citations_must_match_inline_markers(self) -> "GroundedAnswer":
        """Require unique contiguous citations and reject dangling references."""

        citation_numbers = [citation.number for citation in self.citations]
        if len(citation_numbers) != len(set(citation_numbers)):
            raise ValueError("citation numbers must be unique")

        inline_numbers = {int(number) for number in _CITATION_PATTERN.findall(self.answer)}
        if inline_numbers != set(citation_numbers):
            raise ValueError("inline citation markers must match citation records")
        return self


class RefusalCode(StrEnum):
    """Stable machine-readable reasons for withholding an answer."""

    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    CITATION_VALIDATION_FAILED = "CITATION_VALIDATION_FAILED"
    MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"


class RefusalResponse(DomainModel):
    """A deterministic response returned when no grounded answer is safe."""

    outcome: Literal["refusal"] = "refusal"
    code: RefusalCode
    message: NonEmptyText


QueryResponse = Annotated[GroundedAnswer | RefusalResponse, Field(discriminator="outcome")]
