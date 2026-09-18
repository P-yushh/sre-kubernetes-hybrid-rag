"""Grounded-generation composition and provider adapters."""

from sre_rag.generation.base import (
    AllProvidersUnavailableError,
    GenerationBackend,
    GenerationError,
    GenerationRequest,
    InvalidGenerationOutputError,
    ProviderUnavailableError,
)
from sre_rag.generation.fallback import FallbackGenerationBackend
from sre_rag.generation.ollama import OllamaConfig, OllamaGenerationBackend
from sre_rag.generation.openai import OpenAIConfig, OpenAIGenerationBackend
from sre_rag.generation.prompting import GroundedDraftGenerator, GroundedGenerationConfig

__all__ = [
    "AllProvidersUnavailableError",
    "FallbackGenerationBackend",
    "GenerationBackend",
    "GenerationError",
    "GenerationRequest",
    "GroundedDraftGenerator",
    "GroundedGenerationConfig",
    "InvalidGenerationOutputError",
    "OllamaConfig",
    "OllamaGenerationBackend",
    "OpenAIConfig",
    "OpenAIGenerationBackend",
    "ProviderUnavailableError",
]
