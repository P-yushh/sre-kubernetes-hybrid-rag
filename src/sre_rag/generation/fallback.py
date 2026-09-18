"""Operational failover between hosted and local generation providers."""

from sre_rag.domain.generation import GenerationResult
from sre_rag.generation.base import (
    AllProvidersUnavailableError,
    GenerationBackend,
    GenerationRequest,
    ProviderUnavailableError,
)


class FallbackGenerationBackend:
    """Use a fallback only when the primary provider is operationally unavailable."""

    def __init__(self, primary: GenerationBackend, fallback: GenerationBackend) -> None:
        self._primary = primary
        self._fallback = fallback

    def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            return self._primary.generate(request)
        except ProviderUnavailableError:
            try:
                return self._fallback.generate(request)
            except ProviderUnavailableError as fallback_error:
                raise AllProvidersUnavailableError(
                    "primary and fallback generation providers are unavailable"
                ) from fallback_error
