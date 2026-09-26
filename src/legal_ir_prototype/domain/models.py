from __future__ import annotations

"""Domain values exchanged across retrieval pipeline boundaries."""

import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping


def _required_text(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _finite(value: Any, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class SparseQueryVector:
    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("sparse indices and values must have equal length")
        if any(index < 0 for index in self.indices):
            raise ValueError("sparse indices must be non-negative")
        if tuple(sorted(set(self.indices))) != self.indices:
            raise ValueError("sparse indices must be sorted and unique")
        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("sparse values must be finite")


@dataclass(frozen=True, slots=True)
class LegalChunk:
    chunk_id: str
    document_id: str
    passage: str
    retrieval_text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    retrieval_scores: Mapping[str, float] = field(default_factory=dict)
    retrieval_ranks: Mapping[str, int] = field(default_factory=dict)
    fusion_score: float | None = None
    fusion_rank: int | None = None
    reranker_score: float | None = None
    reranker_rank: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", _required_text(self.chunk_id, "chunk_id"))
        object.__setattr__(
            self,
            "document_id",
            _required_text(self.document_id, "document_id"),
        )
        object.__setattr__(self, "passage", _required_text(self.passage, "passage"))
        object.__setattr__(
            self,
            "retrieval_text",
            _required_text(self.retrieval_text, "retrieval_text"),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(
            self,
            "retrieval_scores",
            MappingProxyType(
                {
                    str(channel): _finite(score, f"retrieval_scores.{channel}")
                    for channel, score in self.retrieval_scores.items()
                }
            ),
        )
        normalized_ranks = {
            str(channel): int(rank)
            for channel, rank in self.retrieval_ranks.items()
        }
        if any(rank <= 0 for rank in normalized_ranks.values()):
            raise ValueError("retrieval ranks must be positive")
        object.__setattr__(
            self,
            "retrieval_ranks",
            MappingProxyType(normalized_ranks),
        )
        for field_name in ("fusion_rank", "reranker_rank"):
            rank = getattr(self, field_name)
            if rank is not None and rank <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in ("fusion_score", "reranker_score"):
            score = getattr(self, field_name)
            if score is not None:
                object.__setattr__(self, field_name, _finite(score, field_name))

    @property
    def model_text(self) -> str:
        return self.retrieval_text or self.passage

    def with_scores(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        retrieval_scores: Mapping[str, float] | None = None,
        retrieval_ranks: Mapping[str, int] | None = None,
        fusion_score: float | None = None,
        fusion_rank: int | None = None,
        reranker_score: float | None = None,
        reranker_rank: int | None = None,
    ) -> "LegalChunk":
        return replace(
            self,
            metadata=self.metadata if metadata is None else metadata,
            retrieval_scores=(
                self.retrieval_scores
                if retrieval_scores is None
                else retrieval_scores
            ),
            retrieval_ranks=(
                self.retrieval_ranks if retrieval_ranks is None else retrieval_ranks
            ),
            fusion_score=(
                self.fusion_score if fusion_score is None else fusion_score
            ),
            fusion_rank=self.fusion_rank if fusion_rank is None else fusion_rank,
            reranker_score=(
                self.reranker_score if reranker_score is None else reranker_score
            ),
            reranker_rank=(
                self.reranker_rank if reranker_rank is None else reranker_rank
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "passage": self.passage,
            "retrieval_text": self.retrieval_text,
            "metadata": dict(self.metadata),
            "retrieval_scores": dict(self.retrieval_scores),
            "retrieval_ranks": dict(self.retrieval_ranks),
            "fusion_score": self.fusion_score,
            "fusion_rank": self.fusion_rank,
            "reranker_score": self.reranker_score,
            "reranker_rank": self.reranker_rank,
        }


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    chunk: LegalChunk
    channel: str
    score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel", _required_text(self.channel, "channel"))
        object.__setattr__(self, "score", _finite(self.score, "score"))
