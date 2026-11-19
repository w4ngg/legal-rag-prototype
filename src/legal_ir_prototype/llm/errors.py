"""Normalized errors raised by LLM provider adapters."""


class LLMError(RuntimeError):
    """Base error for the LLM boundary."""


class LLMConfigurationError(LLMError):
    """The configured provider cannot be initialized safely."""


class LLMTransportError(LLMError):
    """The provider could not be reached or returned an HTTP error."""

    def __init__(self, message: str, *, status_code: int | None = None,
                 retryable: bool = True) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class LLMResponseError(LLMError):
    """The provider response envelope was missing required data."""


class LLMValidationError(LLMResponseError):
    """The generated content did not match the legal-answer schema."""


class LLMRefusalError(LLMResponseError):
    """The provider explicitly refused to generate an answer."""
