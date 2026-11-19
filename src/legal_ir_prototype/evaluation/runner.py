from __future__ import annotations

"""Sequential, resumable execution of frozen QA evaluation cases."""

import asyncio
import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from ..api.observability import (
    begin_request_timings,
    end_request_timings,
    observe_request_stage,
    request_timings_snapshot,
)
from ..api.runtime import ApplicationRuntime
from ..core import AppConfig
from ..rag import RAGExecution
from ..rag.prompt_builder import LEGAL_QA_SYSTEM_PROMPT
from ..llm.transport import RetryPolicy
from .dataset import sha256_file
from .recorder import ArtifactRecorder, atomic_write_json, read_jsonl
from .schemas import (
    EvalCase,
    EvaluationError,
    EvaluationPrediction,
    PredictionPayload,
)
from .summary import build_summary
from .trace import EvaluationTrace, Pricing, trace_case

LOGGER = logging.getLogger(__name__)


def _checkpoint_summary(recorder: ArtifactRecorder, count: int,
                        trace: EvaluationTrace | None) -> dict[str, Any]:
    previous = _load_existing_manifest(recorder.summary_path) or {}
    judge_path = previous.get("judge_results_path")
    judges = read_jsonl(Path(judge_path)) if judge_path else []
    summary = build_summary(requested_case_count=count, predictions=recorder.predictions(),
                            errors=recorder.errors(), judge_results=judges)
    if judge_path:
        summary["judge_results_path"] = judge_path
        if "judge_provider_trace" in previous:
            summary["judge_provider_trace"] = previous["judge_provider_trace"]
    if trace is not None:
        summary["provider_trace"] = trace.summary()
    recorder.write_summary(summary)
    return summary


class RAGExecutor(Protocol):
    async def execute(self, query: str) -> RAGExecution:
        ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _milliseconds(
    execution: RAGExecution,
    component_timings: Mapping[str, float],
) -> dict[str, float]:
    timings: dict[str, float] = {
        ("retrieval_total" if stage == "retrieval" else str(stage)): (
            float(duration) * 1000.0
        )
        for stage, duration in component_timings.items()
    }
    for stage, duration in execution.timings_seconds.items():
        output_stage = "retrieval_total" if stage == "retrieval" else stage
        timings[output_stage] = float(duration) * 1000.0
    return {stage: round(duration, 3) for stage, duration in sorted(timings.items())}


def _error_milliseconds(
    *,
    component_timings: Mapping[str, float],
    total_seconds: float,
) -> dict[str, float]:
    timings = {
        str(stage): round(float(duration) * 1000.0, 3)
        for stage, duration in component_timings.items()
    }
    timings["total"] = round(total_seconds * 1000.0, 3)
    return dict(sorted(timings.items()))


def prediction_from_execution(
    *,
    run_id: str,
    request_id: str,
    case: EvalCase,
    execution: RAGExecution,
    component_timings: Mapping[str, float] | None = None,
    include_prompt: bool,
    include_raw_llm_response: bool,
) -> EvaluationPrediction:
    response = execution.response
    llm_answer = execution.llm_answer
    prompt = execution.prompt
    return EvaluationPrediction(
        run_id=run_id,
        request_id=request_id,
        case_id=case.case_id,
        source_query_id=case.source_query_id,
        question=case.question,
        reference_answer=case.reference_answer,
        prediction=PredictionPayload(
            answer=response.answer,
            citations=[value.to_dict() for value in response.citations],
            abstained=response.abstained,
            abstention_reason=response.abstention_reason,
            warnings=list(response.warnings),
            model=response.model,
            input_tokens=None if llm_answer is None else llm_answer.input_tokens,
            output_tokens=None if llm_answer is None else llm_answer.output_tokens,
            finish_reason=None if llm_answer is None else llm_answer.finish_reason,
        ),
        retrieved_chunks=[value.to_dict() for value in execution.retrieved_chunks],
        context=execution.context.to_dict(),
        timings_ms=_milliseconds(execution, component_timings or {}),
        prompt=(
            None
            if not include_prompt or prompt is None
            else {
                "system_prompt": prompt.system_prompt,
                "user_prompt": prompt.user_prompt,
            }
        ),
        raw_llm_response=(
            None
            if not include_raw_llm_response or llm_answer is None
            else llm_answer.raw_response
        ),
        tags=case.tags, source_file=case.source_file,
        expected_behavior=case.expected_behavior,
    )


async def execute_cases(
    *,
    rag_service: RAGExecutor,
    cases: Sequence[EvalCase],
    recorder: ArtifactRecorder,
    run_id: str,
    include_prompt: bool,
    include_raw_llm_response: bool,
    trace: EvaluationTrace | None = None,
) -> None:
    for case in cases:
        if case.case_id in recorder.completed_case_ids:
            continue
        attempt = recorder.attempt_number(case.case_id)
        request_id = f"eval_{run_id}_{case.case_id}_attempt{attempt}"
        token = begin_request_timings()
        started = time.perf_counter()
        try:
            with trace_case(case.case_id, request_id):
                execution = await rag_service.execute(case.question)
            components = request_timings_snapshot()
            prediction = prediction_from_execution(
                run_id=run_id,
                request_id=request_id,
                case=case,
                execution=execution,
                component_timings=components,
                include_prompt=include_prompt,
                include_raw_llm_response=include_raw_llm_response,
            )
        except Exception as exc:
            components = request_timings_snapshot()
            recorder.write_error(
                EvaluationError(
                    run_id=run_id,
                    request_id=request_id,
                    case_id=case.case_id,
                    source_query_id=case.source_query_id,
                    question=case.question,
                    reference_answer=case.reference_answer,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    timings_ms=_error_milliseconds(
                        component_timings=components,
                        total_seconds=time.perf_counter() - started,
                    ),
                )
            )
        else:
            recorder.write_prediction(prediction)
        finally:
            end_request_timings(token)
        summary = _checkpoint_summary(recorder, len(cases), trace)
        if recorder.manifest_path.exists():
            manifest = _load_existing_manifest(recorder.manifest_path)
            manifest.update(updated_at=_utc_now(), completed_case_count=summary["completed_case_count"])
            recorder.write_manifest(manifest)
        LOGGER.info("Evaluation %s/%s case=%s outcome=%s", summary["completed_case_count"],
                    len(cases), case.case_id,
                    "success" if case.case_id in {p["case_id"] for p in recorder.predictions()} else "error")


def _load_existing_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"run manifest must be an object: {path}")
    return value


async def run_evaluation(
    *,
    config_path: Path,
    dataset_path: Path,
    cases: Sequence[EvalCase],
    output_dir: Path,
    resume: bool,
    retry_errors: bool,
    include_prompt: bool,
    include_raw_llm_response: bool,
    require_cuda: bool = False,
    retry_policy: RetryPolicy | None = None,
    request_timeout_seconds: float | None = None,
    pricing: Pricing | None = None,
) -> dict[str, Any]:
    if request_timeout_seconds is not None and (
        not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0
    ):
        raise ValueError("request_timeout_seconds must be finite and positive")
    config_path = config_path.resolve()
    dataset_path = dataset_path.resolve()
    recorder = ArtifactRecorder(
        output_dir,
        resume=resume,
        retry_errors=retry_errors,
    )
    existing = _load_existing_manifest(recorder.manifest_path)
    if existing is None and (recorder.predictions() or recorder.errors()):
        raise ValueError("cannot resume results without their run_manifest.json")
    run_id = str(existing.get("run_id")) if existing else output_dir.name
    config_sha256 = sha256_file(config_path)
    dataset_sha256 = sha256_file(dataset_path)
    prompt_sha256 = hashlib.sha256(LEGAL_QA_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    if existing:
        expected = {
            "config_sha256": config_sha256,
            "dataset_sha256": dataset_sha256,
            "include_prompt": include_prompt,
            "include_raw_llm_response": include_raw_llm_response,
            "requested_case_count": len(cases),
        }
        if "prompt_sha256" in existing:
            expected["prompt_sha256"] = prompt_sha256
        mismatches = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(f"resume configuration does not match run manifest: {mismatches}")
        if "case_ids" in existing and existing["case_ids"] != [c.case_id for c in cases]:
            raise ValueError("resume case selection/order differs from run manifest")

    allowed_ids = {case.case_id for case in cases}
    if any(value["case_id"] not in allowed_ids for value in recorder.predictions() + recorder.errors()):
        raise ValueError("run artifacts contain case IDs outside this dataset selection")
    trace = EvaluationTrace(output_dir / "generation_trace.jsonl", stage="generation",
                            pricing=pricing or Pricing(), resume=resume)
    if existing and allowed_ids <= recorder.completed_case_ids:
        summary = _checkpoint_summary(recorder, len(cases), trace)
        existing.update(status="completed", updated_at=_utc_now(),
                        completed_case_count=summary["completed_case_count"])
        recorder.write_manifest(existing)
        return summary

    config = AppConfig.from_yaml(config_path)
    if request_timeout_seconds is not None and config.llm is not None:
        config = replace(config, llm=replace(config.llm, timeout_seconds=request_timeout_seconds))
    effective = asdict(config)
    effective.pop("source_path")
    snapshot_path = output_dir / "runtime_config.snapshot.json"
    if not existing or not snapshot_path.exists():
        atomic_write_json(snapshot_path, json.loads(json.dumps(effective, default=str)))
    runtime = ApplicationRuntime.build(config, require_cuda=require_cuda,
                                       llm_retry_policy=retry_policy or RetryPolicy(),
                                       llm_attempt_observer=trace.observe)
    runtime.retrieval_service.set_timing_observer(observe_request_stage)
    runtime.rag_service.set_timing_observer(observe_request_stage)
    manifest: dict[str, Any] = dict(existing or {})
    manifest.update(
        {
            "schema_version": 1,
            "run_id": run_id,
            "status": "running",
            "started_at": manifest.get("started_at", _utc_now()),
            "updated_at": _utc_now(),
            "runtime_config": str(config_path),
            "config_sha256": config_sha256,
            "prompt_sha256": prompt_sha256,
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_sha256,
            "requested_case_count": len(cases),
            "case_ids": [c.case_id for c in cases],
            "runtime_snapshot": str(snapshot_path.resolve()),
            "runtime_snapshot_sha256": sha256_file(snapshot_path),
            "retry_policy": asdict(retry_policy or RetryPolicy()),
            "request_timeout_seconds": config.llm.timeout_seconds,
            "pricing": asdict(pricing or Pricing()),
            "include_prompt": include_prompt,
            "include_raw_llm_response": include_raw_llm_response,
            "concurrency": 1,
            "retry_errors_last_invocation": retry_errors,
            "versions": runtime.version_info(),
        }
    )
    recorder.write_manifest(manifest)
    try:
        await asyncio.to_thread(runtime.startup)
        if require_cuda and not runtime.readiness().cuda:
            raise RuntimeError("CUDA is required for this evaluation run")
        await execute_cases(
            rag_service=runtime.rag_service,
            cases=cases,
            recorder=recorder,
            run_id=run_id,
            include_prompt=include_prompt,
            include_raw_llm_response=include_raw_llm_response,
            trace=trace,
        )
    except BaseException as exc:
        manifest.update({"status": "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "failed",
                         "updated_at": _utc_now()})
        summary = _checkpoint_summary(recorder, len(cases), trace)
        manifest["completed_case_count"] = summary["completed_case_count"]
        recorder.write_manifest(manifest)
        raise
    finally:
        await runtime.aclose()
    summary = _checkpoint_summary(recorder, len(cases), trace)
    manifest.update(
        {
            "status": "completed",
            "updated_at": _utc_now(),
            "completed_at": _utc_now(),
            "completed_case_count": summary["completed_case_count"],
        }
    )
    recorder.write_manifest(manifest)
    return summary
