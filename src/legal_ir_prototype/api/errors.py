from __future__ import annotations

"""Stable public API errors without internal exception details."""


class APIError(RuntimeError):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = code
        self.message = message
        self.retryable = bool(retryable)
