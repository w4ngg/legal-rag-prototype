from __future__ import annotations

"""Request-scoped latency collection for structured completion logs."""

from contextvars import ContextVar, Token
from typing import Mapping


_REQUEST_TIMINGS: ContextVar[dict[str, float] | None] = ContextVar(
    "legal_rag_request_timings",
    default=None,
)


def begin_request_timings() -> Token[dict[str, float] | None]:
    return _REQUEST_TIMINGS.set({})


def observe_request_stage(stage: str, duration_seconds: float) -> None:
    timings = _REQUEST_TIMINGS.get()
    if timings is None:
        return
    normalized_stage = str(stage).strip()
    duration = float(duration_seconds)
    if not normalized_stage or duration < 0:
        return
    timings[normalized_stage] = timings.get(normalized_stage, 0.0) + duration


def request_timings_snapshot() -> Mapping[str, float]:
    timings = _REQUEST_TIMINGS.get()
    return {} if timings is None else dict(timings)


def end_request_timings(token: Token[dict[str, float] | None]) -> None:
    _REQUEST_TIMINGS.reset(token)
