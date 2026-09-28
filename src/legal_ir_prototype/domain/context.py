from __future__ import annotations

"""Domain values produced while constructing grounded LLM context."""

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


def _required_text(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _optional_rank(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive")
    return normalized


def _optional_score(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


class ContextExclusionReason(str, Enum):
    """Stable reason codes for chunks omitted from an LLM context."""

    DUPLICATE_CHUNK = "duplicate_chunk"
    DOCUMENT_CHUNK_CAP = "document_chunk_cap"
    MAX_CHUNKS = "max_chunks"
    TOKEN_BUDGET = "token_budget"


@dataclass(frozen=True, slots=True)
class ContextSource:
    """One citable evidence source made from one or more legal chunks."""

    source_id: str
    document_id: str
    chunk_ids: tuple[str, ...]
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    reranker_rank: int | None = None
    reranker_score: float | None = None
    fusion_rank: int | None = None
    fusion_score: float | None = None
    token_count: int = 0

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
        object.__setattr__(self, "text", _required_text(self.text, "text"))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(
            self,
            "reranker_rank",
            _optional_rank(self.reranker_rank, "reranker_rank"),
        )
        object.__setattr__(
            self,
            "fusion_rank",
            _optional_rank(self.fusion_rank, "fusion_rank"),
        )
        object.__setattr__(
            self,
            "reranker_score",
            _optional_score(self.reranker_score, "reranker_score"),
        )
        object.__setattr__(
            self,
            "fusion_score",
            _optional_score(self.fusion_score, "fusion_score"),
        )
        token_count = int(self.token_count)
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        object.__setattr__(self, "token_count", token_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "document_id": self.document_id,
            "chunk_ids": list(self.chunk_ids),
            "text": self.text,
            "metadata": dict(self.metadata),
            "reranker_rank": self.reranker_rank,
            "reranker_score": self.reranker_score,
            "fusion_rank": self.fusion_rank,
            "fusion_score": self.fusion_score,
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class ExcludedChunk:
    """A retrieved chunk omitted from context with an auditable reason."""

    chunk_id: str
    document_id: str
    reason: ContextExclusionReason
    detail: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "chunk_id",
            _required_text(self.chunk_id, "chunk_id"),
        )
        object.__setattr__(
            self,
            "document_id",
            _required_text(self.document_id, "document_id"),
        )
        if not isinstance(self.reason, ContextExclusionReason):
            object.__setattr__(self, "reason", ContextExclusionReason(self.reason))
        if self.detail is not None:
            normalized = str(self.detail).strip()
            object.__setattr__(self, "detail", normalized or None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "reason": self.reason.value,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class BuiltContext:
    """Rendered legal evidence and its lossless source mapping."""

    query: str
    rendered_text: str
    sources: tuple[ContextSource, ...]
    excluded_chunks: tuple[ExcludedChunk, ...]
    total_tokens: int
    truncated: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _required_text(self.query, "query"))
        object.__setattr__(
            self,
            "rendered_text",
            _required_text(self.rendered_text, "rendered_text"),
        )
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "excluded_chunks", tuple(self.excluded_chunks))
        source_ids = [source.source_id for source in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("context source IDs must be unique")
        total_tokens = int(self.total_tokens)
        if total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        object.__setattr__(self, "total_tokens", total_tokens)

    @property
    def selected_chunk_count(self) -> int:
        return sum(len(source.chunk_ids) for source in self.sources)

    @property
    def distinct_document_count(self) -> int:
        return len({source.document_id for source in self.sources})

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "rendered_text": self.rendered_text,
            "sources": [source.to_dict() for source in self.sources],
            "excluded_chunks": [
                excluded.to_dict() for excluded in self.excluded_chunks
            ],
            "total_tokens": self.total_tokens,
            "selected_chunk_count": self.selected_chunk_count,
            "distinct_document_count": self.distinct_document_count,
            "truncated": self.truncated,
        }
