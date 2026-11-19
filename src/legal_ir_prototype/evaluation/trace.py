from __future__ import annotations

"""Durable per-HTTP-attempt telemetry; never record prompts or credentials."""

import math
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .recorder import append_jsonl, read_jsonl


_CASE: ContextVar[dict[str, Any]] = ContextVar("evaluation_trace_case", default={})


@contextmanager
def trace_case(case_id: str, request_id: str):
    token = _CASE.set({"case_id": case_id, "request_id": request_id})
    try:
        yield
    finally:
        _CASE.reset(token)


@contextmanager
def trace_samples(case_ids: list[str], request_id: str):
    token = _CASE.set({"case_ids": case_ids, "request_id": request_id,
                       "samples_per_request": len(case_ids)})
    try:
        yield
    finally:
        _CASE.reset(token)


@dataclass(frozen=True)
class Pricing:
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None

    def __post_init__(self) -> None:
        values = (self.input_usd_per_million, self.output_usd_per_million)
        if (values[0] is None) != (values[1] is None):
            raise ValueError("provide both input and output prices, or neither")
        if any(v is not None and (not math.isfinite(v) or v < 0) for v in values):
            raise ValueError("token prices must be finite and non-negative")

    def estimate(self, usage: Mapping[str, Any] | None) -> float | None:
        if self.input_usd_per_million is None or usage is None:
            return None
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if not isinstance(prompt, int) or not isinstance(completion, int):
            return None
        return round((prompt * self.input_usd_per_million
                      + completion * self.output_usd_per_million) / 1_000_000, 10)


class EvaluationTrace:
    def __init__(self, path: Path, *, stage: str, pricing: Pricing, resume: bool):
        self.path, self.stage, self.pricing = path, stage, pricing
        self.invocation_id = uuid.uuid4().hex
        if path.exists() and not resume:
            raise FileExistsError(f"trace exists; use --continue or a new output: {path}")
        read_jsonl(path, repair_tail=resume)

    def observe(self, event: Mapping[str, Any]) -> None:
        append_jsonl(self.path, {
            "event": "provider_attempt", "timestamp": datetime.now(timezone.utc).isoformat(),
            "invocation_id": self.invocation_id,
            "stage": self.stage, **_CASE.get(), **event,
            "cost_estimate_usd": self.pricing.estimate(event.get("usage")),
            "pricing": asdict(self.pricing),
        })

    def summary(self) -> dict[str, Any]:
        rows = read_jsonl(self.path)
        known = [r["cost_estimate_usd"] for r in rows if r.get("cost_estimate_usd") is not None]
        return {
            "attempt_count": len(rows),
            "successful_http_attempt_count": sum(r.get("outcome") == "success" for r in rows),
            "failed_http_attempt_count": sum(r.get("outcome") != "success" for r in rows),
            "known_response_cost_estimate_usd": round(sum(known), 10) if known else None,
            "costed_attempt_count": len(known),
            "reported_tokens_all_attempts": {
                name: sum(int(r.get("usage", {}).get(key) or 0) for r in rows)
                for name, key in (("input", "prompt_tokens"), ("output", "completion_tokens"))
            },
            "pricing": asdict(self.pricing),
            "cost_scope": "estimate_from_reported_tokens_only; excludes unreported/billed failed requests and cache/tier adjustments",
        }
