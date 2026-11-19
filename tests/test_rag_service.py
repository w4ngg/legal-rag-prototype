from __future__ import annotations

import asyncio

from legal_ir_prototype.domain import LegalChunk, LLMAnswer
from legal_ir_prototype.llm import FakeLLMClient
from legal_ir_prototype.rag import (
    ContextBuildConfig,
    ContextBuilder,
    PromptBuilder,
    RAGService,
)


class FakeRetrieval:
    def __init__(self, chunks: list[LegalChunk]) -> None:
        self.chunks = chunks
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[LegalChunk]:
        self.queries.append(query)
        return self.chunks


def _chunk() -> LegalChunk:
    return LegalChunk(
        "chunk-1",
        "doc-1",
        "Người lao động được nghỉ hằng năm.",
        "Điều 113. Người lao động được nghỉ hằng năm.",
        metadata={"legal": {"articles": ["Điều 113"]}},
        reranker_score=4.2,
        reranker_rank=1,
    )


def _service(retrieval: FakeRetrieval, llm: FakeLLMClient) -> RAGService:
    return RAGService(
        retrieval_service=retrieval,
        context_builder=ContextBuilder(
            ContextBuildConfig(max_context_tokens=2_000)
        ),
        prompt_builder=PromptBuilder(),
        llm_client=llm,
    )


def test_rag_service_returns_only_validated_server_side_citations() -> None:
    retrieval = FakeRetrieval([_chunk()])
    llm = FakeLLMClient(
        LLMAnswer(
            "Người lao động được nghỉ hằng năm.",
            ("SOURCE_1",),
            "fake-model",
            confidence="high",
        )
    )

    response = asyncio.run(_service(retrieval, llm).answer("  câu hỏi  "))

    assert response.abstained is False
    assert response.query == "câu hỏi"
    assert response.citations[0].document_id == "doc-1"
    assert response.citations[0].chunk_ids == ("chunk-1",)
    assert retrieval.queries == ["câu hỏi"]
    assert "SOURCE_1" in llm.calls[0].user_prompt


def test_rag_service_abstains_on_invalid_citation() -> None:
    llm = FakeLLMClient(
        LLMAnswer("Câu trả lời không an toàn.", ("SOURCE_99",), "fake-model")
    )

    response = asyncio.run(
        _service(FakeRetrieval([_chunk()]), llm).answer("query")
    )

    assert response.abstained is True
    assert response.abstention_reason == "citation_validation_failed"
    assert response.citations == ()
    assert response.answer != llm.response.answer
    assert "unknown_citation_ids=SOURCE_99" in response.warnings


def test_rag_service_skips_llm_when_retrieval_returns_no_evidence() -> None:
    llm = FakeLLMClient(
        LLMAnswer("Không được dùng.", ("SOURCE_1",), "fake-model")
    )

    response = asyncio.run(_service(FakeRetrieval([]), llm).answer("query"))

    assert response.abstained is True
    assert response.abstention_reason == "insufficient_context_sources"
    assert response.model == "fake-model"
    assert llm.calls == []


def test_rag_service_execution_exposes_context_and_exact_prompt() -> None:
    llm = FakeLLMClient(
        LLMAnswer("Trả lời.", ("SOURCE_1",), "fake-model")
    )

    execution = asyncio.run(
        _service(FakeRetrieval([_chunk()]), llm).execute("query")
    )

    assert execution.response.abstained is False
    assert execution.retrieved_chunks[0].chunk_id == "chunk-1"
    assert execution.llm_answer is llm.response
    assert execution.context.sources[0].source_id == "SOURCE_1"
    assert execution.prompt is not None
    assert execution.prompt.user_prompt == llm.calls[0].user_prompt
