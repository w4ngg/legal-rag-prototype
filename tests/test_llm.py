from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from legal_ir_prototype.core import LLMConfig
from legal_ir_prototype.domain import LLMAnswer
from legal_ir_prototype.llm import (
    FakeLLMClient,
    LLMClient,
    LLMRefusalError,
    LLMValidationError,
    OpenAICompatibleLLMClient,
    parse_structured_output,
)


def _config(*, mode: str = "json_schema") -> LLMConfig:
    return LLMConfig(
        provider="openai_compatible",
        model="legal-generation-model",
        base_url="https://llm.example.test/v1",
        api_key_env="LEGAL_RAG_LLM_API_KEY",
        timeout_seconds=30.0,
        max_retries=0,
        temperature=0.0,
        max_output_tokens=512,
        structured_output_mode=mode,
    )


def test_pydantic_parser_normalizes_fenced_json_and_rejects_extra_fields() -> None:
    parsed = parse_structured_output(
        """```json
{"answer":" Câu trả lời ","citation_ids":["SOURCE_1","SOURCE_1"],"confidence":"high","insufficient_evidence":false}
```"""
    )

    assert parsed.answer == "Câu trả lời"
    assert parsed.citation_ids == ["SOURCE_1"]
    with pytest.raises(LLMValidationError):
        parse_structured_output(
            '{"answer":"x","citation_ids":[],"confidence":"low",'
            '"insufficient_evidence":true,"unexpected":1}'
        )


def test_fake_client_satisfies_protocol_and_records_prompts() -> None:
    response = LLMAnswer(
        answer="Câu trả lời giả lập.",
        citation_ids=("SOURCE_1",),
        model="fake-model",
        confidence="high",
    )
    client = FakeLLMClient(response)

    result = asyncio.run(
        client.generate(
            system_prompt="Chỉ trả JSON.",
            user_prompt="Câu hỏi và context.",
        )
    )

    assert isinstance(client, LLMClient)
    assert result is response
    assert client.calls[0].system_prompt == "Chỉ trả JSON."


def test_openai_compatible_client_sends_strict_schema_and_parses_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEGAL_RAG_LLM_API_KEY", "secret-test-key")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "legal-generation-model-snapshot",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "Theo SOURCE_1, người lao động được nghỉ.",
                                    "citation_ids": ["SOURCE_1"],
                                    "confidence": "high",
                                    "insufficient_evidence": False,
                                },
                                ensure_ascii=False,
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 30},
            },
        )

    async def scenario() -> LLMAnswer:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = OpenAICompatibleLLMClient(
                _config(),
                http_client=http_client,
            )
            return await client.generate(
                system_prompt="Sinh JSON theo schema.",
                user_prompt="Câu hỏi và SOURCE_1.",
            )

    answer = asyncio.run(scenario())
    payload = captured["payload"]
    assert isinstance(payload, dict)
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert captured["authorization"] == "Bearer secret-test-key"
    assert answer.citation_ids == ("SOURCE_1",)
    assert answer.insufficient_evidence is False
    assert answer.input_tokens == 120
    assert answer.output_tokens == 30


def test_openai_compatible_client_surfaces_provider_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEGAL_RAG_LLM_API_KEY", "secret-test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": None, "refusal": "Không thể trả lời"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = OpenAICompatibleLLMClient(
                _config(),
                http_client=http_client,
            )
            await client.generate(system_prompt="system", user_prompt="user")

    with pytest.raises(LLMRefusalError, match="Không thể trả lời"):
        asyncio.run(scenario())
