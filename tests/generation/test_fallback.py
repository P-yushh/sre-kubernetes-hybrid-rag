"""Tests for hosted-to-local generation failover."""

import pytest

from sre_rag.domain.generation import GenerationProvider
from sre_rag.generation.base import (
    AllProvidersUnavailableError,
    GenerationRequest,
    InvalidGenerationOutputError,
    ProviderUnavailableError,
)
from sre_rag.generation.fallback import FallbackGenerationBackend
from tests.generation.helpers import SequenceBackend, make_generation_result


def _request() -> GenerationRequest:
    return GenerationRequest("query", "instructions", "prompt", {}, 100)


def test_primary_result_is_returned_without_calling_fallback() -> None:
    primary = SequenceBackend((make_generation_result(),))
    fallback = SequenceBackend((make_generation_result(GenerationProvider.OLLAMA),))

    result = FallbackGenerationBackend(primary, fallback).generate(_request())

    assert result.provider is GenerationProvider.OPENAI
    assert primary.calls == 1
    assert fallback.calls == 0


def test_provider_unavailability_uses_local_fallback() -> None:
    primary = SequenceBackend((ProviderUnavailableError("offline"),))
    fallback = SequenceBackend((make_generation_result(GenerationProvider.OLLAMA),))

    result = FallbackGenerationBackend(primary, fallback).generate(_request())

    assert result.provider is GenerationProvider.OLLAMA
    assert fallback.calls == 1


def test_invalid_primary_output_does_not_trigger_fallback() -> None:
    primary = SequenceBackend((InvalidGenerationOutputError("unsafe"),))
    fallback = SequenceBackend((make_generation_result(GenerationProvider.OLLAMA),))

    with pytest.raises(InvalidGenerationOutputError):
        FallbackGenerationBackend(primary, fallback).generate(_request())

    assert fallback.calls == 0


def test_dual_provider_failure_raises_terminal_error() -> None:
    primary = SequenceBackend((ProviderUnavailableError("offline"),))
    fallback = SequenceBackend((ProviderUnavailableError("offline"),))

    with pytest.raises(AllProvidersUnavailableError):
        FallbackGenerationBackend(primary, fallback).generate(_request())
