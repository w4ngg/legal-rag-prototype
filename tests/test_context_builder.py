from __future__ import annotations

from dataclasses import dataclass

from legal_ir_prototype.domain import ContextExclusionReason, LegalChunk
from legal_ir_prototype.rag import ContextBuildConfig, ContextBuilder


@dataclass(frozen=True)
class CharacterCounter:
    def count(self, text: str) -> int:
        return len(text)


def _chunk(
    chunk_id: str,
    document_id: str,
    passage: str,
    *,
    rank: int,
    chunk_index: int | None = None,
    metadata: dict[str, object] | None = None,
) -> LegalChunk:
    values = dict(metadata or {})
    if chunk_index is not None:
        values["chunk_index"] = chunk_index
    return LegalChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        passage=passage,
        retrieval_text=passage,
        metadata=values,
        fusion_score=1.0 / rank,
        fusion_rank=rank,
        reranker_score=10.0 - rank,
        reranker_rank=rank,
    )


def test_builder_preserves_rank_and_renders_legal_metadata() -> None:
    chunks = [
        _chunk(
            "c-1",
            "doc-1",
            "Người lao động được nghỉ hằng năm.",
            rank=1,
            metadata={
                "document_name": "Bộ luật Lao động",
                "legal": {
                    "document_number": "45/2019/QH14",
                    "articles": ["Điều 113"],
                    "clauses": ["Khoản 1"],
                },
            },
        ),
        _chunk("c-2", "doc-2", "Nội dung thứ hai.", rank=2),
    ]

    context = ContextBuilder(
        ContextBuildConfig(max_context_tokens=10_000),
        token_counter=CharacterCounter(),
    ).build(query="Quy định nghỉ hằng năm?", chunks=chunks)

    assert [source.source_id for source in context.sources] == [
        "SOURCE_1",
        "SOURCE_2",
    ]
    assert [source.chunk_ids for source in context.sources] == [
        ("c-1",),
        ("c-2",),
    ]
    assert context.sources[0].reranker_score == 9.0
    assert "Tên văn bản: Bộ luật Lao động" in context.rendered_text
    assert "Số hiệu: 45/2019/QH14" in context.rendered_text
    assert "Điều: Điều 113" in context.rendered_text
    assert "Khoản: Khoản 1" in context.rendered_text
    assert context.total_tokens == len(context.rendered_text)
    assert context.truncated is False


def test_builder_reports_duplicate_document_cap_and_global_cap() -> None:
    first = _chunk("c-1", "doc-1", "Một", rank=1)
    chunks = [
        first,
        first,
        _chunk("c-2", "doc-1", "Hai", rank=2),
        _chunk("c-3", "doc-2", "Ba", rank=3),
        _chunk("c-4", "doc-3", "Bốn", rank=4),
    ]
    context = ContextBuilder(
        ContextBuildConfig(
            max_context_tokens=10_000,
            max_chunks=2,
            max_chunks_per_document=1,
        ),
        token_counter=CharacterCounter(),
    ).build(query="query", chunks=chunks)

    assert [source.chunk_ids for source in context.sources] == [
        ("c-1",),
        ("c-3",),
    ]
    assert [item.reason for item in context.excluded_chunks] == [
        ContextExclusionReason.DUPLICATE_CHUNK,
        ContextExclusionReason.DOCUMENT_CHUNK_CAP,
        ContextExclusionReason.MAX_CHUNKS,
    ]
    assert context.selected_chunk_count == 2
    assert context.distinct_document_count == 2
    assert context.truncated is True


def test_builder_enforces_budget_and_reuses_source_id_after_skip() -> None:
    first = _chunk("c-1", "doc-1", "Ngắn", rank=1)
    first_only = ContextBuilder(
        ContextBuildConfig(
            max_context_tokens=10_000,
            max_chunks=None,
            max_chunks_per_document=None,
        ),
        token_counter=CharacterCounter(),
    ).build(query="query", chunks=[first])
    budget = first_only.total_tokens
    chunks = [
        first,
        _chunk("c-2", "doc-2", "Nội dung quá dài " * 30, rank=2),
        _chunk("c-3", "doc-3", "X", rank=3),
    ]

    context = ContextBuilder(
        ContextBuildConfig(
            max_context_tokens=budget + 70,
            max_chunks=None,
            max_chunks_per_document=None,
        ),
        token_counter=CharacterCounter(),
    ).build(query="query", chunks=chunks)

    assert [source.source_id for source in context.sources] == [
        "SOURCE_1",
        "SOURCE_2",
    ]
    assert [source.chunk_ids for source in context.sources] == [
        ("c-1",),
        ("c-3",),
    ]
    assert context.excluded_chunks[-1].chunk_id == "c-2"
    assert (
        context.excluded_chunks[-1].reason
        == ContextExclusionReason.TOKEN_BUDGET
    )
    assert context.total_tokens <= budget + 70


def test_builder_optionally_merges_adjacent_chunks_and_removes_overlap() -> None:
    chunks = [
        _chunk(
            "c-2",
            "doc-1",
            "Khoản hai. Điều tiếp theo.",
            rank=1,
            chunk_index=2,
            metadata={"legal": {"clauses": ["Khoản 2"]}},
        ),
        _chunk(
            "c-1",
            "doc-1",
            "Khoản một. Khoản hai.",
            rank=2,
            chunk_index=1,
            metadata={"legal": {"clauses": ["Khoản 1"]}},
        ),
    ]

    context = ContextBuilder(
        ContextBuildConfig(
            max_context_tokens=10_000,
            merge_adjacent_chunks=True,
        ),
        token_counter=CharacterCounter(),
    ).build(query="query", chunks=chunks)

    assert len(context.sources) == 1
    assert context.sources[0].chunk_ids == ("c-1", "c-2")
    assert context.sources[0].text == "Khoản một. Khoản hai. Điều tiếp theo."
    assert "Khoản: Khoản 1; Khoản 2" in context.rendered_text
    assert context.sources[0].reranker_rank == 1
