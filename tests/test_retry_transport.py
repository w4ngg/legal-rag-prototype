from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

from legal_ir_prototype.core import LLMConfig
from legal_ir_prototype.evaluation.judge import OpenAICompatibleJudge
from legal_ir_prototype.evaluation.trace import EvaluationTrace, Pricing, trace_case
from legal_ir_prototype.llm import LLMTransportError, OpenAICompatibleLLMClient
from legal_ir_prototype.llm.transport import RetryPolicy, RetryingHTTPTransport, retry_after_seconds


class Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


def test_transient_errors_retry_with_provider_delay_and_pacing():
    clock, events, calls = Clock(), [], []

    def handler(request):
        calls.append(clock.now)
        if len(calls) == 1:
            return httpx.Response(429, json=[{"error": {"details": [{
                "@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12.5s"
            }]}}])
        if len(calls) == 2:
            raise httpx.ReadTimeout("timeout", request=request)
        if len(calls) == 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"usage": {"prompt_tokens": 10, "completion_tokens": 2}})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transport = RetryingHTTPTransport(
                RetryPolicy(max_retries=3, base_delay_seconds=1, request_interval_seconds=5),
                clock=clock, sleep=clock.sleep, observer=events.append,
            )
            return await transport.post(client, "https://example.test/chat", headers={},
                                        payload={"model": "fake"}, timeout_seconds=10)

    assert asyncio.run(scenario()).status_code == 200
    assert len(calls) == 4
    assert clock.waits[0] >= 12.5
    assert all(b - a >= 5 for a, b in zip(calls, calls[1:]))
    assert [e["outcome"] for e in events] == ["http_error", "transport_error", "http_error", "success"]
    assert events[-1]["usage"]["prompt_tokens"] == 10


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_permanent_errors_are_not_retried(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await RetryingHTTPTransport(RetryPolicy()).post(
                client, "https://example.test/chat", headers={}, payload={}, timeout_seconds=1)

    with pytest.raises(LLMTransportError) as error:
        asyncio.run(scenario())
    assert error.value.status_code == status
    assert error.value.retryable is False
    assert len(calls) == 1


def test_retry_budget_does_not_shorten_provider_retry_after():
    clock, calls = Clock(), []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "3600"})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await RetryingHTTPTransport(
                RetryPolicy(budget_seconds=30), clock=clock, sleep=clock.sleep).post(
                client, "https://example.test/chat", headers={}, payload={}, timeout_seconds=10)

    with pytest.raises(LLMTransportError, match="budget exhausted"):
        asyncio.run(scenario())
    assert len(calls) == 1
    assert clock.waits == []


def test_real_async_timeout_is_bounded_and_retried():
    calls = []

    async def handler(request):
        calls.append(request)
        await asyncio.Event().wait()

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await RetryingHTTPTransport(RetryPolicy(max_retries=1, base_delay_seconds=0)).post(
                client, "https://example.test/chat", headers={}, payload={}, timeout_seconds=0.005)

    with pytest.raises(LLMTransportError, match="TimeoutError"):
        asyncio.run(scenario())
    assert len(calls) == 2


def test_retry_after_http_date():
    date = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30))
    assert 28 <= retry_after_seconds(httpx.Response(429, headers={"Retry-After": date})) <= 30


@pytest.mark.parametrize("kind", ["generation", "judge"])
def test_both_clients_use_retry_transport_and_durable_usage_trace(kind, tmp_path):
    calls = []
    trace = EvaluationTrace(tmp_path / "trace.jsonl", stage=kind,
                            pricing=Pricing(1.0, 2.0), resume=False)
    config = LLMConfig(provider="openai_compatible", model="fake", base_url="https://example.test/v1",
                       api_key_env=None, timeout_seconds=1, max_retries=0, temperature=0,
                       max_output_tokens=100, structured_output_mode="json_schema")
    content = ({"answer": "Không đủ thông tin", "citation_ids": [], "confidence": "low", "insufficient_evidence": True}
               if kind == "generation" else {"correctness": 5, "completeness": 5, "faithfulness": 5,
                                             "citation_support": 5, "legal_usefulness": 5,
                                             "overall_pass": True, "rationale": "Phù hợp rubric."})

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        })

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            cls = OpenAICompatibleLLMClient if kind == "generation" else OpenAICompatibleJudge
            adapter = cls(config, http_client=client,
                          retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0),
                          attempt_observer=trace.observe)
            with trace_case("case-1", "req-1"):
                if kind == "generation":
                    return await adapter.generate(system_prompt="system", user_prompt="user")
                from test_evaluation import _case, _rag_service
                from legal_ir_prototype.evaluation.runner import prediction_from_execution
                prediction = prediction_from_execution(run_id="r", request_id="req-1", case=_case("case-1"),
                                                        execution=await _rag_service().execute("q"),
                                                        include_prompt=False, include_raw_llm_response=False)
                return await adapter.evaluate(prediction)

    asyncio.run(scenario())
    summary = trace.summary()
    assert len(calls) == 2
    assert summary["attempt_count"] == 2
    assert summary["known_response_cost_estimate_usd"] == pytest.approx(0.00014)
    assert summary["costed_attempt_count"] == 1
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert all(row["case_id"] == "case-1" and row["request_id"] == "req-1" for row in rows)
    assert all("question" not in row and "headers" not in row for row in rows)
    assert Pricing().estimate({"prompt_tokens": 100, "completion_tokens": 20}) is None
