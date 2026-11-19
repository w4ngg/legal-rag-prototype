from __future__ import annotations

"""Bounded provider retries shared by generation and offline judging."""

import asyncio
import math
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable, Mapping

import httpx

from .errors import LLMTransportError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 5
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 60.0
    budget_seconds: float = 300.0
    request_interval_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        for name in ("base_delay_seconds", "max_delay_seconds", "budget_seconds",
                     "request_interval_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.budget_seconds <= 0:
            raise ValueError("budget_seconds must be positive")


def retry_after_seconds(response: httpx.Response) -> float:
    """Honor both HTTP Retry-After and Google's google.rpc.RetryInfo."""
    delays = [0.0]
    header = response.headers.get("retry-after")
    if header:
        try:
            delays.append(float(header))
        except ValueError:
            try:
                date = parsedate_to_datetime(header)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                delays.append((date - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    try:
        body = response.json()
        bodies = body if isinstance(body, list) else [body]
        for item in bodies:
            error = item.get("error", {}) if isinstance(item, dict) else {}
            if not isinstance(error, dict):
                continue
            for detail in error.get("details", []):
                if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("RetryInfo"):
                    delays.append(float(str(detail.get("retryDelay", "0s")).removesuffix("s")))
    except (ValueError, TypeError, AttributeError):
        pass
    return max(value for value in delays if math.isfinite(value))


class RetryingHTTPTransport:
    def __init__(self, policy: RetryPolicy, *,
                 observer: Callable[[Mapping[str, Any]], None] | None = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.policy = policy
        self.observer = observer
        self.sleep = sleep
        self.clock = clock
        self._last_started: float | None = None

    async def post(self, client: httpx.AsyncClient, endpoint: str, *,
                   headers: Mapping[str, str], payload: Mapping[str, Any],
                   timeout_seconds: float) -> httpx.Response:
        policy = self.policy
        deadline = self.clock() + policy.budget_seconds
        delay = 0.0
        last_error = LLMTransportError("LLM retry budget exhausted")
        for attempt in range(1, policy.max_retries + 2):
            pacing = (0.0 if self._last_started is None else max(
                0.0, self._last_started + policy.request_interval_seconds - self.clock()))
            wait = max(delay, pacing)
            if self.clock() + wait >= deadline:
                raise LLMTransportError(
                    f"LLM retry budget exhausted: {last_error}",
                    status_code=last_error.status_code,
                    retryable=last_error.retryable,
                ) from last_error
            if wait:
                await self.sleep(wait)
            started = self.clock()
            if started >= deadline:
                raise LLMTransportError("LLM retry budget exhausted before next request") from last_error
            self._last_started = started
            response: httpx.Response | None = None
            event: dict[str, Any] = {"attempt": attempt, "model": payload.get("model"),
                                     "wait_seconds": round(wait, 3)}
            try:
                response = await asyncio.wait_for(
                    client.post(endpoint, headers=headers, json=payload,
                                timeout=min(timeout_seconds, deadline - started)),
                    timeout=min(timeout_seconds, deadline - started),
                )
                event["status_code"] = response.status_code
                if response.is_success:
                    try:
                        body = response.json()
                        usage = body.get("usage") if isinstance(body, dict) else None
                        if isinstance(usage, dict):
                            event["usage"] = usage
                    except ValueError:
                        pass
                    event["outcome"] = "success"
                    return response
                retryable = response.status_code in {408, 425, 429} or response.status_code >= 500
                last_error = LLMTransportError(
                    f"LLM endpoint returned HTTP {response.status_code}",
                    status_code=response.status_code, retryable=retryable,
                )
                event.update(outcome="http_error", retryable=retryable)
            except (httpx.TransportError, asyncio.TimeoutError) as exc:
                last_error = LLMTransportError(f"LLM request failed: {type(exc).__name__}")
                event.update(outcome="transport_error", error_type=type(exc).__name__, retryable=True)
            finally:
                event.setdefault("outcome", "cancelled")
                event["latency_ms"] = round((self.clock() - started) * 1000, 3)
                if self.observer is not None:
                    self.observer(event)
            if not last_error.retryable or attempt > policy.max_retries:
                raise last_error
            backoff = min(policy.base_delay_seconds * 2 ** min(attempt - 1, 30),
                          policy.max_delay_seconds)
            delay = max(backoff + random.uniform(0, backoff * 0.25),
                        0.0 if response is None else retry_after_seconds(response))
            LOGGER.warning("Provider retry attempt=%s/%s status=%s wait_seconds=%.3f",
                           attempt + 1, policy.max_retries + 1,
                           last_error.status_code, delay)
        raise AssertionError("unreachable retry state")
