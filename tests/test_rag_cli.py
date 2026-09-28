from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_ir_prototype.core import LLMConfig, RAGConfig
from legal_ir_prototype.domain import LegalChunk
from legal_ir_prototype import rag_cli


class FakeConfig:
    def __init__(self, llm: LLMConfig | None) -> None:
        self.llm = llm
        self.rag = RAGConfig()


class FakeRetrieval:
    def retrieve(self, query: str) -> list[LegalChunk]:
        assert query == "Câu hỏi pháp luật?"
        return [
            LegalChunk(
                "chunk-1",
                "doc-1",
                "Nội dung pháp luật.",
                "Điều 1. Nội dung pháp luật.",
                reranker_score=3.0,
                reranker_rank=1,
            )
        ]


def _fake_llm_config() -> LLMConfig:
    return LLMConfig(
        provider="fake",
        model="fake-legal-llm",
        base_url=None,
        api_key_env=None,
        timeout_seconds=30.0,
        max_retries=0,
        temperature=0.0,
        max_output_tokens=256,
        structured_output_mode="json_schema",
    )


def test_rag_smoke_cli_runs_end_to_end_with_fake_provider(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        rag_cli.AppConfig,
        "from_yaml",
        lambda path: FakeConfig(_fake_llm_config()),
    )
    monkeypatch.setattr(rag_cli, "build_service", lambda config: FakeRetrieval())

    exit_code = rag_cli.main(
        [
            "--config",
            "runtime.yaml",
            "--query",
            " Câu hỏi pháp luật? ",
            "--show-context",
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert result["query"] == "Câu hỏi pháp luật?"
    assert result["answer"].startswith("Đây là câu trả lời giả lập")
    assert result["citations"][0]["source_id"] == "SOURCE_1"
    assert result["abstained"] is False
    assert result["debug"]["llm_called"] is True
    assert "<legal_sources>" in result["debug"]["context"]["rendered_text"]
    assert "SOURCE_1" in result["debug"]["llm_prompt"]["user_prompt"]


def test_rag_smoke_cli_requires_llm_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag_cli.AppConfig,
        "from_yaml",
        lambda path: FakeConfig(None),
    )

    with pytest.raises(SystemExit, match="must include an llm section"):
        rag_cli.main(
            ["--config", str(Path("runtime.yaml")), "--query", "query"]
        )
