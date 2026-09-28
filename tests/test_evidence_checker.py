from __future__ import annotations

from legal_ir_prototype.domain import LegalChunk, LLMAnswer
from legal_ir_prototype.rag import (
    CitationValidator,
    ContextBuildConfig,
    ContextBuilder,
    EvidenceChecker,
    EvidencePolicy,
)


def _context(*scores: float):
    chunks = [
        LegalChunk(
            f"chunk-{index}",
            f"doc-{index}",
            f"Nội dung {index}.",
            f"Nội dung {index}.",
            reranker_score=score,
            reranker_rank=index,
        )
        for index, score in enumerate(scores, start=1)
    ]
    return ContextBuilder(ContextBuildConfig(max_context_tokens=1_000)).build(
        query="query",
        chunks=chunks,
    )


def test_checker_rejects_empty_or_uncalibrated_low_score_context() -> None:
    empty = _context()
    assert EvidenceChecker().check_context(empty).reason == (
        "insufficient_context_sources"
    )

    checker = EvidenceChecker(
        EvidencePolicy(min_top_reranker_score=2.0, min_top_score_margin=0.5)
    )
    assert checker.check_context(_context(1.5)).reason == (
        "top_reranker_score_below_threshold"
    )
    assert checker.check_context(_context(3.0, 2.8)).reason == (
        "top_reranker_margin_below_threshold"
    )
    assert checker.check_context(_context(3.0, 2.0)).sufficient


def test_checker_rejects_invalid_citations_and_llm_abstention() -> None:
    context = _context(3.0)
    invalid_answer = LLMAnswer("Trả lời.", ("SOURCE_9",), "fake")
    invalid = CitationValidator().validate(
        answer=invalid_answer,
        context=context,
    )
    decision = EvidenceChecker().check_answer(
        answer=invalid_answer,
        citations=invalid,
    )
    assert decision.reason == "citation_validation_failed"
    assert "unknown_citation_ids=SOURCE_9" in decision.warnings

    abstained_answer = LLMAnswer(
        "Không đủ căn cứ.",
        (),
        "fake",
        insufficient_evidence=True,
    )
    valid = CitationValidator().validate(
        answer=abstained_answer,
        context=context,
    )
    assert EvidenceChecker().check_answer(
        answer=abstained_answer,
        citations=valid,
    ).reason == "llm_reported_insufficient_evidence"
