from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from legal_ir_prototype.domain import LegalChunk, LLMAnswer
from legal_ir_prototype.evaluation.dataset import load_cases, prepare_dataset
from legal_ir_prototype.evaluation.deterministic import run_deterministic_checks
from legal_ir_prototype.evaluation.judge import (
    JudgeOutput,
    judge_predictions,
)
from legal_ir_prototype.evaluation.recorder import ArtifactRecorder, read_jsonl
from legal_ir_prototype.evaluation.runner import (
    execute_cases,
    prediction_from_execution,
)
from legal_ir_prototype.evaluation.schemas import (
    EvalCase,
    EvaluationPrediction,
    JudgeAssessment,
)
from legal_ir_prototype.llm import FakeLLMClient
from legal_ir_prototype.rag import ContextBuilder, RAGService


class FakeRetrieval:
    def retrieve(self, query: str) -> list[LegalChunk]:
        return [
            LegalChunk(
                chunk_id=f"chunk-{query}",
                document_id="doc-1",
                passage="Người lao động được nghỉ hằng năm.",
                retrieval_text="Điều 113. Người lao động được nghỉ hằng năm.",
                reranker_score=4.2,
                reranker_rank=1,
            )
        ]


def _rag_service() -> RAGService:
    return RAGService(
        retrieval_service=FakeRetrieval(),
        context_builder=ContextBuilder(),
        llm_client=FakeLLMClient(
            LLMAnswer(
                answer="Người lao động được nghỉ hằng năm.",
                citation_ids=("SOURCE_1",),
                model="fake-model",
                input_tokens=100,
                output_tokens=20,
                finish_reason="stop",
                raw_response='{"answer":"raw"}',
            )
        ),
    )


def _case(case_id: str, question: str = "Câu hỏi") -> EvalCase:
    return EvalCase(
        case_id=case_id,
        source_query_id=case_id.removeprefix("qa_"),
        question=question,
        reference_answer="Câu trả lời tham chiếu.",
    )


def test_prepare_dataset_is_reproducible_and_records_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "train.json"
    source.write_text(
        json.dumps(
            {
                str(index): {"question": f"q{index}", "answer": f"a{index}"}
                for index in range(20)
            }
        ),
        encoding="utf-8",
    )
    first = prepare_dataset(
        source=source,
        output_dir=tmp_path / "first",
        sample_size=5,
        seed=2026,
    )
    second = prepare_dataset(
        source=source,
        output_dir=tmp_path / "second",
        sample_size=5,
        seed=2026,
    )

    assert first.selected_source_query_ids == second.selected_source_query_ids
    assert first.source_sha256 == second.source_sha256
    assert load_cases(tmp_path / "first" / "cases.jsonl") == load_cases(
        tmp_path / "second" / "cases.jsonl"
    )
    with pytest.raises(FileExistsError):
        prepare_dataset(
            source=source,
            output_dir=tmp_path / "first",
            sample_size=5,
            seed=2026,
        )


def test_eval_case_schema_rejects_empty_question() -> None:
    with pytest.raises(ValidationError):
        EvalCase(
            case_id="qa_1",
            source_query_id="1",
            question=" ",
            reference_answer="answer",
        )


def test_prediction_serializes_trace_and_respects_sensitive_flags() -> None:
    execution = asyncio.run(_rag_service().execute("query"))
    prediction = prediction_from_execution(
        run_id="run1",
        request_id="req1",
        case=_case("qa_1"),
        execution=execution,
        component_timings={"reranker": 0.012, "qdrant_dense": 0.003},
        include_prompt=False,
        include_raw_llm_response=False,
    )

    assert prediction.prediction.input_tokens == 100
    assert prediction.prediction.finish_reason == "stop"
    assert prediction.retrieved_chunks[0]["chunk_id"] == "chunk-query"
    assert prediction.timings_ms["reranker"] == 12.0
    assert prediction.timings_ms["qdrant_dense"] == 3.0
    assert "retrieval_total" in prediction.timings_ms
    assert prediction.prompt is None
    assert prediction.raw_llm_response is None
    assert run_deterministic_checks(prediction).passed is True


class SometimesFailingService:
    def __init__(self, *, fail_bad: bool = True) -> None:
        self.delegate = _rag_service()
        self.calls: list[str] = []
        self.fail_bad = fail_bad

    async def execute(self, query: str):
        self.calls.append(query)
        if query == "bad" and self.fail_bad:
            raise RuntimeError("expected failure")
        return await self.delegate.execute(query)


def test_runner_continues_after_error_and_resume_skips_completed(tmp_path: Path) -> None:
    cases = (_case("qa_1", "good"), _case("qa_2", "bad"), _case("qa_3", "good2"))
    first_service = SometimesFailingService()
    recorder = ArtifactRecorder(tmp_path / "run", resume=False)
    asyncio.run(
        execute_cases(
            rag_service=first_service,
            cases=cases,
            recorder=recorder,
            run_id="run",
            include_prompt=False,
            include_raw_llm_response=False,
        )
    )

    assert len(read_jsonl(recorder.predictions_path)) == 2
    assert len(read_jsonl(recorder.errors_path)) == 1
    assert first_service.calls == ["good", "bad", "good2"]

    resumed_service = SometimesFailingService()
    resumed = ArtifactRecorder(tmp_path / "run", resume=True)
    asyncio.run(
        execute_cases(
            rag_service=resumed_service,
            cases=cases,
            recorder=resumed,
            run_id="run",
            include_prompt=False,
            include_raw_llm_response=False,
        )
    )
    assert resumed_service.calls == []

    retry_service = SometimesFailingService(fail_bad=False)
    retrying = ArtifactRecorder(
        tmp_path / "run",
        resume=True,
        retry_errors=True,
    )
    asyncio.run(
        execute_cases(
            rag_service=retry_service,
            cases=cases,
            recorder=retrying,
            run_id="run",
            include_prompt=False,
            include_raw_llm_response=False,
        )
    )
    assert retry_service.calls == ["bad"]
    assert len(retrying.predictions()) == 3
    assert retrying.errors() == []
    assert len(read_jsonl(retrying.errors_path)) == 1
    retried = [
        value
        for value in retrying.predictions()
        if value["case_id"] == "qa_2"
    ][0]
    assert retried["request_id"].endswith("_attempt2")


class FakeJudge:
    model = "fake-judge"

    async def evaluate(self, prediction: EvaluationPrediction) -> JudgeOutput:
        return JudgeOutput(
            assessment=JudgeAssessment(
                correctness=5,
                completeness=4,
                faithfulness=5,
                citation_support=5,
                legal_usefulness=4,
                overall_pass=True,
                rationale="Đúng và có căn cứ.",
            ),
            model=self.model,
            input_tokens=200,
            output_tokens=50,
        )

    async def aclose(self) -> None:
        return None


class FailingJudge(FakeJudge):
    async def evaluate(self, prediction: EvaluationPrediction) -> JudgeOutput:
        raise RuntimeError("temporary provider failure")


def test_fake_judge_writes_resumable_structured_result(tmp_path: Path) -> None:
    execution = asyncio.run(_rag_service().execute("query"))
    prediction = prediction_from_execution(
        run_id="run1",
        request_id="req1",
        case=_case("qa_1"),
        execution=execution,
        include_prompt=False,
        include_raw_llm_response=False,
    )
    output = tmp_path / "judge_results.jsonl"
    results = asyncio.run(
        judge_predictions(
            predictions=(prediction,),
            output_path=output,
            client=FakeJudge(),
            resume=False,
        )
    )
    resumed = asyncio.run(
        judge_predictions(
            predictions=(prediction,),
            output_path=output,
            client=FakeJudge(),
            resume=True,
        )
    )

    assert results[0].assessment is not None
    assert results[0].assessment.overall_pass is True
    assert results == resumed
    assert len(read_jsonl(output)) == 1


def test_judge_retry_errors_calls_only_failed_cases_and_uses_latest_result(
    tmp_path: Path,
) -> None:
    execution = asyncio.run(_rag_service().execute("query"))
    prediction = prediction_from_execution(
        run_id="run1",
        request_id="req1",
        case=_case("qa_1"),
        execution=execution,
        include_prompt=False,
        include_raw_llm_response=False,
    )
    output = tmp_path / "judge_results.jsonl"
    failed = asyncio.run(
        judge_predictions(
            predictions=(prediction,),
            output_path=output,
            client=FailingJudge(),
            resume=False,
        )
    )
    retried = asyncio.run(
        judge_predictions(
            predictions=(prediction,),
            output_path=output,
            client=FakeJudge(),
            resume=True,
            retry_errors=True,
        )
    )

    assert failed[0].error is not None
    assert retried[0].error is None
    assert retried[0].assessment is not None
    assert len(read_jsonl(output)) == 2
