from __future__ import annotations

import pytest

from legal_ir_prototype.domain import LegalChunk, LLMAnswer
from legal_ir_prototype.rag import (
    CitationValidationError,
    CitationValidator,
    ContextBuildConfig,
    ContextBuilder,
)


def _context():
    chunk = LegalChunk(
        "chunk-1",
        "doc-1",
        "Nội dung Điều 113.",
        "Nội dung Điều 113.",
        metadata={"legal": {"articles": ["Điều 113"]}},
        reranker_score=4.2,
        reranker_rank=1,
    )
    return ContextBuilder(ContextBuildConfig(max_context_tokens=1_000)).build(
        query="Quy định nghỉ hằng năm?",
        chunks=[chunk],
    )


def test_validator_resolves_only_server_owned_sources_and_metadata() -> None:
    answer = LLMAnswer(
        answer="Được nghỉ hằng năm.",
        citation_ids=("SOURCE_1", "SOURCE_1"),
        model="fake-model",
    )

    result = CitationValidator().validate(answer=answer, context=_context())

    assert result.is_valid
    assert len(result.citations) == 1
    assert result.citations[0].document_id == "doc-1"
    assert result.citations[0].chunk_ids == ("chunk-1",)
    assert result.citations[0].metadata["legal"]["articles"] == ["Điều 113"]


def test_validator_rejects_unknown_and_missing_citations() -> None:
    context = _context()
    unknown = LLMAnswer("Trả lời.", ("SOURCE_99",), "fake-model")
    missing = LLMAnswer("Trả lời.", (), "fake-model")

    unknown_result = CitationValidator().validate(
        answer=unknown,
        context=context,
    )
    missing_result = CitationValidator().validate(
        answer=missing,
        context=context,
    )

    assert unknown_result.invalid_source_ids == ("SOURCE_99",)
    assert unknown_result.missing_required_citations is True
    assert missing_result.missing_required_citations is True
    with pytest.raises(CitationValidationError):
        CitationValidator().validate_or_raise(answer=unknown, context=context)


def test_validator_allows_no_citation_for_explicit_insufficient_evidence() -> None:
    answer = LLMAnswer(
        answer="Không đủ căn cứ.",
        citation_ids=(),
        model="fake-model",
        insufficient_evidence=True,
    )

    result = CitationValidator().validate(answer=answer, context=_context())

    assert result.is_valid
    assert result.citations == ()
