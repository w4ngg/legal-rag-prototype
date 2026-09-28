from __future__ import annotations

import pytest

from legal_ir_prototype.domain import Citation, LLMAnswer, RAGResponse


def test_llm_answer_keeps_provider_output_untrusted_and_hides_raw_by_default() -> None:
    answer = LLMAnswer(
        answer="Người lao động được nghỉ hằng năm.",
        citation_ids=("SOURCE_1", "UNKNOWN_SOURCE"),
        model="fake-legal-llm",
        confidence="high",
        input_tokens=100,
        output_tokens=20,
        raw_response='{"answer":"raw"}',
    )

    assert "UNKNOWN_SOURCE" in answer.citation_ids
    assert "raw_response" not in answer.to_dict()
    assert answer.to_dict(include_raw_response=True)["raw_response"] is not None


def test_rag_response_requires_consistent_abstention_state() -> None:
    citation = Citation("SOURCE_1", "doc-1", ("chunk-1",), {"article": "113"})
    response = RAGResponse(
        query="Quy định nghỉ hằng năm?",
        answer="Căn cứ nguồn đã cung cấp.",
        citations=(citation,),
        model="fake-legal-llm",
    )

    assert response.to_dict()["citations"][0]["source_id"] == "SOURCE_1"
    with pytest.raises(ValueError, match="abstention_reason is required"):
        RAGResponse(
            query="query",
            answer="Không đủ căn cứ.",
            citations=(),
            model="fake-legal-llm",
            abstained=True,
        )
