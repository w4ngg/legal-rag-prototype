"""Normalized errors raised by LLM provider adapters."""


class LLMError(RuntimeError):
    """Base error for the LLM boundary."""


class LLMConfigurationError(LLMError):
    """The configured provider cannot be initialized safely."""


class LLMTransportError(LLMError):
    """The provider could not be reached or returned an HTTP error."""


class LLMResponseError(LLMError):
    """The provider response envelope was missing required data."""


class LLMValidationError(LLMResponseError):
    """The generated content did not match the legal-answer schema."""


class LLMRefusalError(LLMResponseError):
    """The provider explicitly refused to generate an answer."""
