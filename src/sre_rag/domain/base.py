"""Shared types and validation behavior for domain contracts."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Revision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{7,64}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class DomainModel(BaseModel):
    """Reject unknown data and prevent mutation after validation."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
