from __future__ import annotations

"""Deterministic construction of citable legal evidence for an LLM."""

import html
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from ..domain.context import (
    BuiltContext,
    ContextExclusionReason,
    ContextSource,
    ExcludedChunk,
)
from ..domain.models import LegalChunk


class TokenCounter(Protocol):
    """Model-specific token counting boundary used by ContextBuilder."""

    def count(self, text: str) -> int:
        """Return the number of input tokens represented by ``text``."""
        ...


@dataclass(frozen=True, slots=True)
class ApproximateTokenCounter:
    """Conservative fallback until an LLM tokenizer is configured."""

    characters_per_token: float = 3.0

    def __post_init__(self) -> None:
        normalized = float(self.characters_per_token)
        if not math.isfinite(normalized):
            raise ValueError("characters_per_token must be finite")
        if normalized <= 0:
            raise ValueError("characters_per_token must be positive")
        object.__setattr__(self, "characters_per_token", normalized)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self.characters_per_token))


@dataclass(frozen=True, slots=True)
class ContextBuildConfig:
    """Selection and rendering policy independent from retrieval settings."""

    max_context_tokens: int = 10_000
    max_chunks: int | None = 8
    max_chunks_per_document: int | None = 2
    include_legal_metadata: bool = True
    merge_adjacent_chunks: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_chunks",
            _positive_or_none(self.max_chunks, "max_chunks"),
        )
        object.__setattr__(
            self,
            "max_chunks_per_document",
            _positive_or_none(
                self.max_chunks_per_document,
                "max_chunks_per_document",
            ),
        )
        if isinstance(self.max_context_tokens, bool):
            raise ValueError("max_context_tokens must be a positive integer")
        max_context_tokens = int(self.max_context_tokens)
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be a positive integer")
        object.__setattr__(self, "max_context_tokens", max_context_tokens)


@dataclass(frozen=True, slots=True)
class _SourceCandidate:
    chunks: tuple[LegalChunk, ...]
    first_position: int
    text: str
    metadata: Mapping[str, Any]


def _positive_or_none(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer or null")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer or null")
    return normalized


def _optional_integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _chunk_order_key(chunk: LegalChunk, fallback_position: int) -> tuple[int, int]:
    chunk_index = _optional_integer(chunk.metadata.get("chunk_index"))
    if chunk_index is not None:
        return chunk_index, fallback_position
    character_start = _optional_integer(chunk.metadata.get("character_start"))
    if character_start is not None:
        return character_start, fallback_position
    return fallback_position, fallback_position


def _are_adjacent(left: LegalChunk, right: LegalChunk) -> bool:
    if left.document_id != right.document_id:
        return False
    left_index = _optional_integer(left.metadata.get("chunk_index"))
    right_index = _optional_integer(right.metadata.get("chunk_index"))
    if left_index is not None and right_index is not None:
        return abs(left_index - right_index) == 1
    left_start = _optional_integer(left.metadata.get("character_start"))
    left_end = _optional_integer(left.metadata.get("character_end"))
    right_start = _optional_integer(right.metadata.get("character_start"))
    right_end = _optional_integer(right.metadata.get("character_end"))
    if None in {left_start, left_end, right_start, right_end}:
        return False
    assert left_start is not None and left_end is not None
    assert right_start is not None and right_end is not None
    earlier_end, later_start = (
        (left_end, right_start)
        if left_start <= right_start
        else (right_end, left_start)
    )
    return later_start <= earlier_end


def _merge_overlapping_text(left: str, right: str) -> str:
    maximum = min(len(left), len(right))
    for overlap in range(maximum, 0, -1):
        if left[-overlap:] == right[:overlap]:
            return left + right[overlap:]
    return f"{left}\n\n{right}"


def _unique_values(*values: Any) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            candidates = value
        elif value is None:
            candidates = ()
        else:
            candidates = (value,)
        for candidate in candidates:
            if candidate not in result:
                result.append(candidate)
    return result


def _merge_metadata(chunks: Sequence[LegalChunk]) -> dict[str, Any]:
    merged = dict(chunks[0].metadata)
    legal: dict[str, Any] = {}
    for chunk in chunks:
        candidate = chunk.metadata.get("legal")
        if not isinstance(candidate, Mapping):
            continue
        for key, value in candidate.items():
            if isinstance(value, (list, tuple, set)):
                legal[key] = _unique_values(legal.get(key), value)
            elif key not in legal or legal[key] is None or legal[key] == "":
                legal[key] = value
    if legal:
        merged["legal"] = legal
    for field_name, operation in (
        ("chunk_index", min),
        ("character_start", min),
        ("character_end", max),
    ):
        values = [
            value
            for chunk in chunks
            if (value := _optional_integer(chunk.metadata.get(field_name)))
            is not None
        ]
        if values:
            merged[field_name] = operation(values)
    return merged


def _candidate_from_group(
    positioned_chunks: Sequence[tuple[int, LegalChunk]],
) -> _SourceCandidate:
    ordered = sorted(
        positioned_chunks,
        key=lambda item: _chunk_order_key(item[1], item[0]),
    )
    chunks = tuple(chunk for _, chunk in ordered)
    text = chunks[0].passage
    for chunk in chunks[1:]:
        text = _merge_overlapping_text(text, chunk.passage)
    return _SourceCandidate(
        chunks=chunks,
        first_position=min(position for position, _ in positioned_chunks),
        text=text,
        metadata=_merge_metadata(chunks),
    )


def _group_candidates(
    chunks: Sequence[LegalChunk],
    *,
    merge_adjacent: bool,
) -> list[_SourceCandidate]:
    if not merge_adjacent:
        return [
            _candidate_from_group(((position, chunk),))
            for position, chunk in enumerate(chunks)
        ]
    groups: list[list[tuple[int, LegalChunk]]] = []
    for position, chunk in enumerate(chunks):
        matching = [
            index
            for index, group in enumerate(groups)
            if group[0][1].document_id == chunk.document_id
            and any(_are_adjacent(existing, chunk) for _, existing in group)
        ]
        if not matching:
            groups.append([(position, chunk)])
            continue
        primary = matching[0]
        groups[primary].append((position, chunk))
        for redundant in reversed(matching[1:]):
            groups[primary].extend(groups.pop(redundant))
    candidates = [_candidate_from_group(group) for group in groups]
    return sorted(candidates, key=lambda candidate: candidate.first_position)


def _best_rank(chunks: Sequence[LegalChunk], field_name: str) -> int | None:
    values = [
        value
        for chunk in chunks
        if (value := getattr(chunk, field_name)) is not None
    ]
    return min(values) if values else None


def _best_score(chunks: Sequence[LegalChunk], field_name: str) -> float | None:
    values = [
        value
        for chunk in chunks
        if (value := getattr(chunk, field_name)) is not None
    ]
    return max(values) if values else None


def _display_values(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        normalized = [str(item).strip() for item in value if str(item).strip()]
        return "; ".join(normalized) or None
    normalized = str(value).strip()
    return normalized or None


def _metadata_lines(
    metadata: Mapping[str, Any],
    *,
    include_legal_metadata: bool,
) -> list[str]:
    fields: list[tuple[str, Any]] = [
        ("Tên văn bản", metadata.get("document_name")),
        ("Tiêu đề truy hồi", metadata.get("retrieval_title")),
    ]
    if include_legal_metadata:
        legal = metadata.get("legal")
        if isinstance(legal, Mapping):
            fields.extend(
                [
                    ("Loại văn bản", legal.get("document_type")),
                    ("Số hiệu", legal.get("document_number")),
                    ("Cơ quan ban hành", legal.get("issuing_authority")),
                    ("Ngày ban hành", legal.get("issued_at")),
                    ("Ngày hiệu lực", legal.get("effective_at")),
                    ("Chương", legal.get("chapters")),
                    ("Mục", legal.get("sections")),
                    ("Điều", legal.get("articles")),
                    ("Khoản", legal.get("clauses")),
                    ("Điểm", legal.get("points")),
                ]
            )
    fields.append(("Nguồn", metadata.get("source_link")))
    lines: list[str] = []
    for label, value in fields:
        rendered = _display_values(value)
        if rendered is not None:
            lines.append(f"{label}: {rendered}")
    return lines


def _render_source(
    source_id: str,
    candidate: _SourceCandidate,
    *,
    include_legal_metadata: bool,
) -> str:
    document_id = html.escape(candidate.chunks[0].document_id, quote=True)
    lines = [
        f'<SOURCE id="{html.escape(source_id, quote=True)}" '
        f'document_id="{document_id}">'
    ]
    lines.extend(
        html.escape(line)
        for line in _metadata_lines(
            candidate.metadata,
            include_legal_metadata=include_legal_metadata,
        )
    )
    if len(lines) > 1:
        lines.append("")
    lines.extend(("Nội dung:", html.escape(candidate.text), "</SOURCE>"))
    return "\n".join(lines)


def _render_context(blocks: Sequence[str]) -> str:
    if not blocks:
        return "<legal_sources>\n</legal_sources>"
    return "<legal_sources>\n" + "\n\n".join(blocks) + "\n</legal_sources>"


class ContextBuilder:
    """Select and render ranked chunks without changing retrieval scores."""

    def __init__(
        self,
        config: ContextBuildConfig | None = None,
        *,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.config = config or ContextBuildConfig()
        self.token_counter = token_counter or ApproximateTokenCounter()

    def build(
        self,
        *,
        query: str,
        chunks: Sequence[LegalChunk],
    ) -> BuiltContext:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        excluded: list[ExcludedChunk] = []
        unique_chunks: list[LegalChunk] = []
        seen: set[tuple[str, str]] = set()
        document_counts: defaultdict[str, int] = defaultdict(int)
        for chunk in chunks:
            key = (chunk.document_id, chunk.chunk_id)
            if key in seen:
                excluded.append(
                    ExcludedChunk(
                        chunk.chunk_id,
                        chunk.document_id,
                        ContextExclusionReason.DUPLICATE_CHUNK,
                    )
                )
                continue
            seen.add(key)
            document_cap = self.config.max_chunks_per_document
            if (
                document_cap is not None
                and document_counts[chunk.document_id] >= document_cap
            ):
                excluded.append(
                    ExcludedChunk(
                        chunk.chunk_id,
                        chunk.document_id,
                        ContextExclusionReason.DOCUMENT_CHUNK_CAP,
                        f"max_chunks_per_document={document_cap}",
                    )
                )
                continue
            if (
                self.config.max_chunks is not None
                and len(unique_chunks) >= self.config.max_chunks
            ):
                excluded.append(
                    ExcludedChunk(
                        chunk.chunk_id,
                        chunk.document_id,
                        ContextExclusionReason.MAX_CHUNKS,
                        f"max_chunks={self.config.max_chunks}",
                    )
                )
                continue
            unique_chunks.append(chunk)
            document_counts[chunk.document_id] += 1

        candidates = _group_candidates(
            unique_chunks,
            merge_adjacent=self.config.merge_adjacent_chunks,
        )
        empty_context = _render_context(())
        if self.token_counter.count(empty_context) > self.config.max_context_tokens:
            raise ValueError("max_context_tokens is too small for context envelope")

        sources: list[ContextSource] = []
        blocks: list[str] = []
        for candidate in candidates:
            source_id = f"SOURCE_{len(sources) + 1}"
            block = _render_source(
                source_id,
                candidate,
                include_legal_metadata=self.config.include_legal_metadata,
            )
            prospective = _render_context((*blocks, block))
            prospective_tokens = self.token_counter.count(prospective)
            if prospective_tokens > self.config.max_context_tokens:
                for chunk in candidate.chunks:
                    excluded.append(
                        ExcludedChunk(
                            chunk.chunk_id,
                            chunk.document_id,
                            ContextExclusionReason.TOKEN_BUDGET,
                            (
                                "candidate would use "
                                f"{prospective_tokens}/"
                                f"{self.config.max_context_tokens} tokens"
                            ),
                        )
                    )
                continue
            sources.append(
                ContextSource(
                    source_id=source_id,
                    document_id=candidate.chunks[0].document_id,
                    chunk_ids=tuple(
                        chunk.chunk_id for chunk in candidate.chunks
                    ),
                    text=candidate.text,
                    metadata=candidate.metadata,
                    reranker_rank=_best_rank(candidate.chunks, "reranker_rank"),
                    reranker_score=_best_score(
                        candidate.chunks,
                        "reranker_score",
                    ),
                    fusion_rank=_best_rank(candidate.chunks, "fusion_rank"),
                    fusion_score=_best_score(candidate.chunks, "fusion_score"),
                    token_count=self.token_counter.count(block),
                )
            )
            blocks.append(block)

        rendered = _render_context(blocks)
        truncation_reasons = {
            ContextExclusionReason.DOCUMENT_CHUNK_CAP,
            ContextExclusionReason.MAX_CHUNKS,
            ContextExclusionReason.TOKEN_BUDGET,
        }
        return BuiltContext(
            query=normalized_query,
            rendered_text=rendered,
            sources=tuple(sources),
            excluded_chunks=tuple(excluded),
            total_tokens=self.token_counter.count(rendered),
            truncated=any(item.reason in truncation_reasons for item in excluded),
        )
