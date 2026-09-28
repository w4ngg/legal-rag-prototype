from __future__ import annotations

"""Pydantic schema shared by prompts and OpenAI-compatible responses."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from .errors import LLMValidationError


class StructuredLegalAnswer(BaseModel):
    """Exact JSON object the generation model must produce."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    citation_ids: list[str]
    confidence: Literal["low", "medium", "high"]
    insufficient_evidence: bool

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("answer must not be empty")
        return normalized

    @field_validator("citation_ids")
    @classmethod
    def normalize_citation_ids(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            source_id = value.strip()
            if not source_id:
                raise ValueError("citation IDs must not be empty")
            if source_id not in normalized:
                normalized.append(source_id)
        return normalized


def structured_output_json_schema() -> dict[str, object]:
    """Return a strict-schema-compatible copy owned by the caller."""

    return json.loads(json.dumps(StructuredLegalAnswer.model_json_schema()))


def _unwrap_json_fence(value: str) -> str:
    normalized = value.strip()
    if not normalized.startswith("```"):
        return normalized
    lines = normalized.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        return normalized
    return "\n".join(lines[1:-1]).strip()


def parse_structured_output(value: str) -> StructuredLegalAnswer:
    raw = _unwrap_json_fence(value)
    if not raw:
        raise LLMValidationError("LLM returned empty structured output")
    try:
        return StructuredLegalAnswer.model_validate_json(raw)
    except ValidationError as exc:
        raise LLMValidationError(
            f"LLM structured output failed validation: {exc}"
        ) from exc
