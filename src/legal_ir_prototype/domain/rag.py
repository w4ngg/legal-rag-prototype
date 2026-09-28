from __future__ import annotations

"""Provider-independent domain values for grounded legal QA."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _required_text(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_non_negative_integer(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


@dataclass(frozen=True, slots=True)
class LLMAnswer:
    """Untrusted structured output returned by an LLM provider adapter."""

    answer: str
    citation_ids: tuple[str, ...]
    model: str
    confidence: str | None = None
    insufficient_evidence: bool = False
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw_response: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "answer", _required_text(self.answer, "answer"))
        citation_ids = tuple(
            _required_text(value, "citation_ids") for value in self.citation_ids
        )
        object.__setattr__(self, "citation_ids", citation_ids)
        object.__setattr__(self, "model", _required_text(self.model, "model"))
        object.__setattr__(self, "confidence", _optional_text(self.confidence))
        object.__setattr__(
            self,
            "insufficient_evidence",
            bool(self.insufficient_evidence),
        )
        object.__setattr__(
            self,
            "finish_reason",
            _optional_text(self.finish_reason),
        )
        object.__setattr__(
            self,
            "input_tokens",
            _optional_non_negative_integer(self.input_tokens, "input_tokens"),
        )
        object.__setattr__(
            self,
            "output_tokens",
            _optional_non_negative_integer(self.output_tokens, "output_tokens"),
        )
        object.__setattr__(self, "raw_response", _optional_text(self.raw_response))

    def to_dict(self, *, include_raw_response: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "answer": self.answer,
            "citation_ids": list(self.citation_ids),
            "model": self.model,
            "confidence": self.confidence,
            "insufficient_evidence": self.insufficient_evidence,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
        if include_raw_response:
            value["raw_response"] = self.raw_response
        return value


@dataclass(frozen=True, slots=True)
class Citation:
    """A citation resolved and validated against a BuiltContext source."""

    source_id: str
    document_id: str
    chunk_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_id",
            _required_text(self.source_id, "source_id"),
        )
        object.__setattr__(
            self,
            "document_id",
            _required_text(self.document_id, "document_id"),
        )
        chunk_ids = tuple(
            _required_text(value, "chunk_ids") for value in self.chunk_ids
        )
        if not chunk_ids:
            raise ValueError("chunk_ids must not be empty")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunk_ids must be unique")
        object.__setattr__(self, "chunk_ids", chunk_ids)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "document_id": self.document_id,
            "chunk_ids": list(self.chunk_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RAGResponse:
    """Validated public response returned by the RAG orchestration layer."""

    query: str
    answer: str
    citations: tuple[Citation, ...]
    model: str
    abstained: bool = False
    abstention_reason: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _required_text(self.query, "query"))
        object.__setattr__(self, "answer", _required_text(self.answer, "answer"))
        object.__setattr__(self, "citations", tuple(self.citations))
        source_ids = [citation.source_id for citation in self.citations]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("citations must have unique source IDs")
        object.__setattr__(self, "model", _required_text(self.model, "model"))
        reason = _optional_text(self.abstention_reason)
        if self.abstained and reason is None:
            raise ValueError("abstention_reason is required when abstained is true")
        if not self.abstained and reason is not None:
            raise ValueError("abstention_reason requires abstained=true")
        object.__setattr__(self, "abstention_reason", reason)
        warnings = tuple(
            _required_text(value, "warnings") for value in self.warnings
        )
        object.__setattr__(self, "warnings", warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "model": self.model,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "warnings": list(self.warnings),
        }
