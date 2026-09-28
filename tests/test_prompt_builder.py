from __future__ import annotations

from legal_ir_prototype.domain import LegalChunk
from legal_ir_prototype.rag import (
    ContextBuildConfig,
    ContextBuilder,
    PromptBuilder,
)


def test_prompt_controls_grounding_citations_and_structured_output() -> None:
    chunk = LegalChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        passage="Nội dung Điều 113.",
        retrieval_text="Nội dung Điều 113.",
        metadata={
            "document_name": "Bo-Luat-lao-dong-2019-333670",
            "retrieval_title": "Bo Luat lao dong 2019 333670",
            "legal": {
                "articles": ["Điều 113"],
                "clauses": ["Khoản 1"],
            },
        },
        reranker_score=4.2,
        reranker_rank=1,
    )
    context = ContextBuilder(
        ContextBuildConfig(max_context_tokens=1_000)
    ).build(query="Quy định nghỉ hằng năm?", chunks=[chunk])

    prompt = PromptBuilder().build(context)

    assert "Chỉ sử dụng bằng chứng" in prompt.system_prompt
    assert "insufficient_evidence" in prompt.system_prompt
    assert "citation_ids" in prompt.system_prompt
    assert "answer phải bắt đầu bằng căn cứ pháp lý" in prompt.system_prompt
    assert '"Luat nghia vu quan su 2015 282383"' in prompt.system_prompt
    assert '"Luật Nghĩa vụ quân sự 2015"' in prompt.system_prompt
    assert "Xóa mã định danh số nằm ở cuối" in prompt.system_prompt
    assert "Không chèn SOURCE_n hoặc URL vào answer" in prompt.system_prompt
    assert "SOURCE_1" in prompt.user_prompt
    assert "Điều 113" in prompt.user_prompt
    assert "Khoản 1" in prompt.user_prompt
    assert "Tiêu đề truy hồi: Bo Luat lao dong 2019 333670" in prompt.user_prompt
    assert context.query in prompt.user_prompt
