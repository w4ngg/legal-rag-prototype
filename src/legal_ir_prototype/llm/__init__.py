"""Provider-independent and OpenAI-compatible LLM clients."""

from .base import FakeLLMClient, LLMCall, LLMClient
from .errors import (
    LLMConfigurationError,
    LLMError,
    LLMRefusalError,
    LLMResponseError,
    LLMTransportError,
    LLMValidationError,
)
from .openai_compatible import OpenAICompatibleLLMClient
from .schemas import StructuredLegalAnswer, parse_structured_output

__all__ = [
    "FakeLLMClient",
    "LLMCall",
    "LLMClient",
    "LLMConfigurationError",
    "LLMError",
    "LLMRefusalError",
    "LLMResponseError",
    "LLMTransportError",
    "LLMValidationError",
    "OpenAICompatibleLLMClient",
    "StructuredLegalAnswer",
    "parse_structured_output",
]
