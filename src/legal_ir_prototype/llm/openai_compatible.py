from __future__ import annotations

"""OpenAI Chat Completions compatible structured-output client."""

import asyncio
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from ..core import LLMConfig
from ..domain import LLMAnswer
from .errors import (
    LLMConfigurationError,
    LLMRefusalError,
    LLMResponseError,
    LLMTransportError,
)
from .schemas import parse_structured_output, structured_output_json_schema


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str | None = None
    refusal: str | None = None


class _Choice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: _Message
    finish_reason: str | None = None


class _Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class _Completion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    choices: list[_Choice]
    usage: _Usage | None = None


class OpenAICompatibleLLMClient:
    """Call a configurable `/chat/completions` endpoint and validate JSON."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if config.provider != "openai_compatible":
            raise LLMConfigurationError(
                "OpenAICompatibleLLMClient requires provider=openai_compatible"
            )
        if config.base_url is None:
            raise LLMConfigurationError("OpenAI-compatible base URL is required")
        api_key = config.api_key()
        if config.api_key_env is not None and api_key is None:
            raise LLMConfigurationError(
                f"missing LLM API key environment variable: {config.api_key_env}"
            )
        self.config = config
        self.endpoint = f"{config.base_url.rstrip('/')}/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        if api_key is not None:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self._client = http_client or httpx.AsyncClient()
        self._owns_client = http_client is None

    @property
    def model(self) -> str:
        return self.config.model

    async def __aenter__(self) -> "OpenAICompatibleLLMClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _response_format(self) -> dict[str, Any]:
        if self.config.structured_output_mode == "json_object":
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "legal_answer",
                "strict": True,
                "schema": structured_output_json_schema(),
            },
        }

    def _payload(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if not system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        if not user_prompt.strip():
            raise ValueError("user_prompt must not be empty")
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "response_format": self._response_format(),
        }

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            try:
                response = await self._client.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
            except httpx.TransportError as exc:
                if attempt + 1 >= attempts:
                    raise LLMTransportError(
                        f"LLM request failed after {attempts} attempt(s): {exc}"
                    ) from exc
                await asyncio.sleep(min(2**attempt, 8))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 < attempts:
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                response_preview = response.text[:500]
                raise LLMTransportError(
                    f"LLM endpoint returned HTTP {response.status_code}: "
                    f"{response_preview}"
                ) from exc
            return response
        raise AssertionError("unreachable LLM retry state")

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMAnswer:
        response = await self._post(
            self._payload(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        )
        try:
            completion = _Completion.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise LLMResponseError(
                "LLM endpoint returned an invalid Chat Completions envelope"
            ) from exc
        if not completion.choices:
            raise LLMResponseError("LLM response contains no choices")
        choice = completion.choices[0]
        if choice.message.refusal:
            raise LLMRefusalError(f"LLM refused the request: {choice.message.refusal}")
        if choice.finish_reason == "length":
            raise LLMResponseError("LLM output was truncated at the token limit")
        if choice.finish_reason == "content_filter":
            raise LLMResponseError("LLM output was interrupted by content filtering")
        if choice.message.content is None:
            raise LLMResponseError("LLM response contains no message content")
        structured = parse_structured_output(choice.message.content)
        usage = completion.usage
        return LLMAnswer(
            answer=structured.answer,
            citation_ids=tuple(structured.citation_ids),
            model=(completion.model or self.config.model),
            confidence=structured.confidence,
            insufficient_evidence=structured.insufficient_evidence,
            finish_reason=choice.finish_reason,
            input_tokens=None if usage is None else usage.prompt_tokens,
            output_tokens=None if usage is None else usage.completion_tokens,
            raw_response=choice.message.content,
        )
