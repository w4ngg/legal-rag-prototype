from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from legal_ir_prototype.core import LLMConfig
from legal_ir_prototype.evaluation.cli import main
from legal_ir_prototype.evaluation.diagnostic_judge import (
    DiagnosticJudge, PromptBatchOutput, SYSTEM_PROMPT, judge_diagnostic_predictions,
    pack_samples, request_usage_summary, sample_payload,
)
from legal_ir_prototype.evaluation.recorder import append_jsonl, read_jsonl
from legal_ir_prototype.llm.transport import RetryPolicy
from test_diagnostic_analysis import assessment, prediction


def config():
    return LLMConfig("openai_compatible", "fake-judge", "https://example.test/v1", None,
                     1, 0, 0, 1024, "json_schema")


def response_for(request, *, missing=None, unknown=False, truncated=False):
    payload = json.loads(request.content)
    samples = json.loads(payload["messages"][1]["content"])["samples"]
    results = [{"case_id": s["case_id"], "assessment": assessment().model_dump(mode="json")}
               for s in reversed(samples) if s["case_id"] != missing]
    for item in results:
        item["assessment"]["rationale"] = f"Assessment of {item['case_id']}"
    if unknown:
        results[0]["case_id"] = "unknown-case"
    return httpx.Response(200, json={
        "choices": [{"message": {"content": json.dumps({"results": results})},
                     "finish_reason": "length" if truncated else "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    })


async def run(values, output, handler, *, resume=False, on_progress=None, **kwargs):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DiagnosticJudge(config(), http_client=http, retry_policy=RetryPolicy(max_retries=0))
        return await judge_diagnostic_predictions(predictions=values, output_path=output, client=client,
                                                  resume=resume, retry_errors=True,
                                                  on_progress=on_progress, **kwargs)


def test_five_samples_share_one_prompt_and_results_are_joined_by_id(tmp_path):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return response_for(request)
    values = [prediction(str(i)) for i in range(11)]
    results = asyncio.run(run(values, tmp_path / "judge.jsonl", handler))
    assert len(results) == 11
    assert [len(json.loads(c["messages"][1]["content"])["samples"]) for c in calls] == [5, 5, 1]
    assert [c["max_tokens"] for c in calls] == [10240, 10240, 2048]
    assert [r.case_id for r in results] == [p.case_id for p in values]
    assert all(r.assessment.rationale == f"Assessment of {r.case_id}" for r in results)
    assert all(r.input_tokens is None and r.output_tokens is None for r in results)
    usage = request_usage_summary([r.model_dump(mode="json") for r in results])
    assert usage["request_count_in_case_artifacts"] == 3
    assert usage["input_total"] == 300 and usage["output_total"] == 60
    assert "sample khác" in SYSTEM_PROMPT and "KHÔNG phải bằng chứng" in SYSTEM_PROMPT
    assert all(not r.diagnostic_validation_errors for r in results)


def test_300_samples_use_60_http_requests_and_per_request_usage(tmp_path):
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        return response_for(request)
    results = asyncio.run(run([prediction(str(i)) for i in range(300)], tmp_path / "judge.jsonl", handler))
    assert len(results) == 300 and calls == 60
    usage = request_usage_summary([r.model_dump(mode="json") for r in results])
    assert usage["input_total"] == 6000
    assert usage["output_total"] == 1200


def test_single_sample_control_mode_uses_same_diagnostic_schema(tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return response_for(request)
    results = asyncio.run(run([prediction(str(i)) for i in range(3)], tmp_path / "judge.jsonl",
                             handler, samples_per_prompt=1))
    assert len(calls) == 3
    assert all(r.assessment.diagnostics.context_sufficiency == "sufficient" for r in results)


def test_missing_one_result_preserves_other_four_and_continue_calls_only_missing(tmp_path):
    values = [prediction(str(i)) for i in range(5)]
    output = tmp_path / "judge.jsonl"
    failed = asyncio.run(run(values, output, lambda request: response_for(request, missing="2")))
    assert [r.case_id for r in failed if r.error] == ["2"]
    calls = []
    def handler(request):
        calls.extend(s["case_id"] for s in json.loads(json.loads(request.content)["messages"][1]["content"])["samples"])
        return response_for(request)
    recovered = asyncio.run(run(values, output, handler, resume=True))
    assert calls == ["2"]
    assert all(r.error is None for r in recovered)
    assert len(read_jsonl(output)) == 6


@pytest.mark.parametrize("bad", ["unknown", "truncated"])
def test_protocol_error_marks_affected_group_failed_not_success(tmp_path, bad):
    results = asyncio.run(run([prediction(str(i)) for i in range(5)], tmp_path / "judge.jsonl",
                             lambda r: response_for(r, unknown=bad == "unknown", truncated=bad == "truncated")))
    assert len(results) == 5 and all(r.error and r.assessment is None for r in results)


def test_one_api_failure_does_not_discard_other_groups_and_retry_skips_successes(tmp_path):
    values = [prediction(str(i)) for i in range(11)]
    output = tmp_path / "judge.jsonl"
    def handler(request):
        samples = json.loads(json.loads(request.content)["messages"][1]["content"])["samples"]
        return httpx.Response(503) if samples[0]["case_id"] == "5" else response_for(request)
    initial = asyncio.run(run(values, output, handler))
    assert sum(bool(r.error) for r in initial) == 5
    seen = []
    def recovered(request):
        seen.extend(s["case_id"] for s in json.loads(json.loads(request.content)["messages"][1]["content"])["samples"])
        return response_for(request)
    result = asyncio.run(run(values, output, recovered, resume=True))
    assert seen == ["5", "6", "7", "8", "9"]
    assert len(result) == 11 and all(r.error is None for r in result)
    assert len(read_jsonl(output)) == 16


def test_interruption_after_two_committed_items_does_not_repeat_them(tmp_path):
    values = [prediction(str(i)) for i in range(11)]
    output = tmp_path / "judge.jsonl"
    count = 0
    def interrupt():
        nonlocal count
        count += 1
        if count == 2:
            raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run(values, output, response_for, on_progress=interrupt))
    assert len(read_jsonl(output)) == 2
    assert json.loads(output.with_suffix(".manifest.json").read_text())["status"] == "interrupted"
    seen = []
    def handler(request):
        seen.extend(s["case_id"] for s in json.loads(json.loads(request.content)["messages"][1]["content"])["samples"])
        return response_for(request)
    results = asyncio.run(run(values, output, handler, resume=True))
    assert seen == [str(i) for i in range(2, 11)]
    assert len(results) == len(read_jsonl(output)) == 11


def test_resume_rejects_changed_grouping_and_changed_predictions(tmp_path):
    output = tmp_path / "judge.jsonl"
    values = [prediction("1")]
    asyncio.run(run(values, output, response_for))
    with pytest.raises(ValueError, match="grouping config changed"):
        asyncio.run(run(values, output, response_for, resume=True, samples_per_prompt=2))
    changed = values[0].model_copy(update={"question": "Câu hỏi đã đổi"})
    with pytest.raises(ValueError, match="predictions changed"):
        asyncio.run(run([changed], output, response_for, resume=True))


def test_prompt_budget_splits_without_truncation_and_rejects_single_oversized_sample(tmp_path):
    values = [prediction(str(i)) for i in range(5)]
    one_size = len(json.dumps(sample_payload(values[0]), ensure_ascii=False)) + 2
    limit = len(SYSTEM_PROMPT) + 20 + one_size * 2
    groups = pack_samples(values, 5, limit)
    assert [len(g) for g in groups] == [2, 2, 1]
    calls = []
    def handler(request):
        calls.append(request)
        return response_for(request)
    results = asyncio.run(run(values[:1], tmp_path / "judge.jsonl", handler, max_prompt_characters=10))
    assert calls == []
    assert "no context was truncated" in results[0].error


def test_duplicate_case_id_does_not_overwrite_another_result(tmp_path):
    def handler(request):
        value = json.loads(response_for(request).content)
        body = json.loads(value["choices"][0]["message"]["content"])
        body["results"].append(body["results"][0])
        value["choices"][0]["message"]["content"] = json.dumps(body)
        return httpx.Response(200, json=value)
    results = asyncio.run(run([prediction(str(i)) for i in range(5)], tmp_path / "judge.jsonl", handler))
    assert sum(bool(r.error) for r in results) == 1
    assert results[-1].error == "duplicate_case_id"


def test_cli_diagnostic_writes_analysis_and_request_cost_without_double_counting(monkeypatch, tmp_path):
    import legal_ir_prototype.evaluation.cli as cli
    original = DiagnosticJudge
    calls = []
    def handler(request):
        calls.append(request)
        return response_for(request)
    def factory(*args, **kwargs):
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = original(*args, http_client=http, **kwargs)
        client._owns_client = True
        return client
    monkeypatch.setattr(cli, "DiagnosticJudge", factory)
    monkeypatch.setenv("LEGAL_RAG_LLM_API_KEY", "fake-test-key")
    source = tmp_path / "predictions.jsonl"
    for i in range(11):
        append_jsonl(source, prediction(str(i)))
    output = tmp_path / "diagnostic.jsonl"
    config_path = Path(__file__).parents[1] / "configs/judge_large.yaml"
    result = main(["judge-diagnostic", "--config", str(config_path), "--predictions", str(source),
                   "--output", str(output), "--env-file", str(tmp_path / "none"), "--continue",
                   "--request-interval", "0", "--input-usd-per-million", "1", "--output-usd-per-million", "2"])
    assert result == 0 and len(calls) == 3
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["judge"]["tokens"] == {"input_total": 300, "output_total": 60}
    assert summary["judge_provider_trace"]["known_response_cost_estimate_usd"] == pytest.approx(0.00042)
    analysis = json.loads((tmp_path / "diagnostic.analysis/summary.json").read_text())
    assert analysis["outcomes"]["answered_correctly"]["count"] == 11
    traces = read_jsonl(output.with_suffix(".trace.jsonl"))
    assert [len(row["case_ids"]) for row in traces] == [5, 5, 1]
