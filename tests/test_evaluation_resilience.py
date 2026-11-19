from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from legal_ir_prototype.evaluation import runner
from legal_ir_prototype.evaluation.cli import _config_from_manifest, main
from legal_ir_prototype.evaluation.dataset import load_cases, prepare_dataset, question_key, sha256_file
from legal_ir_prototype.evaluation.judge import judge_predictions
from legal_ir_prototype.evaluation.recorder import ArtifactRecorder, append_jsonl, artifact_lock, read_jsonl
from legal_ir_prototype.evaluation.summary import build_summary
from legal_ir_prototype.core import AppConfig
from test_evaluation import FakeJudge, SometimesFailingService, _case, _rag_service


def test_resume_recovers_only_truncated_final_record_and_backs_it_up(tmp_path):
    path = tmp_path / "results.jsonl"
    good = b'{"case_id":"1"}\n'
    partial = b'{"case_id":"2", "answer":"dang'
    path.write_bytes(good + partial)
    assert read_jsonl(path, repair_tail=True) == [{"case_id": "1"}]
    assert path.read_bytes() == good
    assert next(tmp_path.glob("*.tail")).read_bytes() == partial
    append_jsonl(path, {"case_id": "2"})
    assert len(read_jsonl(path)) == 2
    path.write_bytes(good + b'broken\n' + good)
    with pytest.raises(ValueError, match="invalid JSON"):
        read_jsonl(path, repair_tail=True)
    assert b'broken' in path.read_bytes()


def test_resume_preserves_complete_record_without_final_newline(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"case_id":"1"}')
    read_jsonl(path, repair_tail=True)
    append_jsonl(path, {"case_id": "2"})
    assert [v["case_id"] for v in read_jsonl(path)] == ["1", "2"]


def test_duplicate_process_lock_is_rejected(tmp_path):
    with artifact_lock(tmp_path / "run.lock"):
        with pytest.raises(RuntimeError, match="another evaluation process"):
            with artifact_lock(tmp_path / "run.lock"):
                pytest.fail("must not acquire the same active output")


def test_multisource_sampling_preserves_answers_and_excludes_null_and_duplicates(tmp_path):
    first, second = tmp_path / "train.json", tmp_path / "warmup.json"
    first.write_text(json.dumps({"1": {"question": "Câu hỏi A", "answer": "Đáp án A"},
                                 "2": {"question": "Câu hỏi B", "answer": None},
                                 "3": {"question": "Câu hỏi C", "answer": "Đáp án C"}}))
    second.write_text(json.dumps({"1": {"question": " CÂU  HỎI A ", "answer": "Đáp án A"},
                                  "4": {"question": "Câu hỏi D", "answer": "Đáp án D"}}))
    manifest = prepare_dataset(source=[first, second], source_counts=[1, 1], sample_size=2,
                               seed=2026, strategy="stratified", output_dir=tmp_path / "sample")
    cases = load_cases(tmp_path / "sample/cases.jsonl")
    assert len(cases) == 2
    assert cases[1].reference_answer == "Đáp án D"
    assert manifest.cases_sha256 == sha256_file(tmp_path / "sample/cases.jsonl")
    assert {c.source_file for c in cases} == {f"{tmp_path.name}/train.json", f"{tmp_path.name}/warmup.json"}
    assert any("missing_labeled" in e.reason for e in manifest.excluded_cases)


class InterruptedService(SometimesFailingService):
    async def execute(self, query):
        if query == "interrupt":
            self.calls.append(query)
            raise asyncio.CancelledError()
        return await super().execute(query)


def test_generation_checkpoint_resume_and_frozen_runtime(monkeypatch, tmp_path):
    config = tmp_path / "runtime.yaml"
    config.write_text((Path(__file__).parents[1] / "configs/runtime.yaml").read_text())
    cases = (_case("1", "good"), _case("2", "interrupt"), _case("3", "good2"))
    dataset = tmp_path / "cases.jsonl"
    for case in cases:
        append_jsonl(dataset, case)
    first = InterruptedService()
    builds = []

    class Runtime:
        def __init__(self, service):
            self.rag_service = service
            self.rag_service.set_timing_observer = lambda observer: None
            self.retrieval_service = SimpleNamespace(set_timing_observer=lambda observer: None)
        def startup(self):
            pass
        def version_info(self):
            return {"llm_model": "fake"}
        async def aclose(self):
            pass

    service = first
    def build(*args, **kwargs):
        builds.append(kwargs)
        return Runtime(service)
    monkeypatch.setattr(runner.ApplicationRuntime, "build", build)
    output = tmp_path / "run"
    kwargs = dict(config_path=config, dataset_path=dataset, cases=cases, output_dir=output,
                  resume=True, retry_errors=True, include_prompt=False, include_raw_llm_response=False)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.run_evaluation(**kwargs))
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["completed_case_count"] == 1
    assert len(read_jsonl(output / "predictions.jsonl")) == 1
    service = SometimesFailingService(fail_bad=False)
    result = asyncio.run(runner.run_evaluation(**kwargs))
    assert service.calls == ["interrupt", "good2"]
    assert result["successful_case_count"] == 3
    assert json.loads((output / "run_manifest.json").read_text())["status"] == "completed"
    before = len(builds)
    asyncio.run(runner.run_evaluation(**kwargs))
    assert len(builds) == before  # Complete runs do not even reload models.
    with monkeypatch.context() as patch:
        patch.setattr(runner, "LEGAL_QA_SYSTEM_PROMPT", "changed prompt")
        with pytest.raises(ValueError, match="resume configuration"):
            asyncio.run(runner.run_evaluation(**kwargs))
    with pytest.raises(ValueError, match="resume configuration"):
        asyncio.run(runner.run_evaluation(**{**kwargs, "cases": cases[:1]}))
    snapshot = _config_from_manifest(output / "predictions.jsonl")
    assert AppConfig.from_yaml(snapshot).llm.model == "gemini-3.1-flash-lite"
    config.write_text(config.read_text().replace("gemini-3.1-flash-lite", "changed-model"))
    assert _config_from_manifest(output / "predictions.jsonl") == snapshot
    with pytest.raises(ValueError, match="resume configuration"):
        asyncio.run(runner.run_evaluation(**kwargs))


def test_cli_continue_enables_resume_and_retry_errors(monkeypatch, tmp_path):
    dataset = tmp_path / "cases.jsonl"
    append_jsonl(dataset, _case("1"))
    captured = {}
    async def fake_run(**kwargs):
        captured.update(kwargs)
        return {"ok": True}
    monkeypatch.setattr("legal_ir_prototype.evaluation.cli.run_evaluation", fake_run)
    assert main(["run", "--config", "unused.yaml", "--dataset", str(dataset),
                 "--output", str(tmp_path / "run"), "--env-file", str(tmp_path / "none"),
                 "--continue", "--request-interval", "0"]) == 0
    assert captured["resume"] is True and captured["retry_errors"] is True


def test_judge_interruption_resume_and_model_change_rejection(tmp_path):
    async def predictions():
        execution = await _rag_service().execute("q")
        return tuple(runner.prediction_from_execution(run_id="r", request_id=f"req-{i}", case=_case(str(i)),
                                                      execution=execution, include_prompt=False,
                                                      include_raw_llm_response=False) for i in range(3))
    values = asyncio.run(predictions())
    calls = []
    class InterruptingJudge(FakeJudge):
        async def evaluate(self, prediction):
            calls.append(prediction.case_id)
            if prediction.case_id == "1":
                raise asyncio.CancelledError()
            return await super().evaluate(prediction)
    output = tmp_path / "judge.jsonl"
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(judge_predictions(predictions=values, output_path=output,
                                      client=InterruptingJudge(), resume=True, retry_errors=True))
    assert len(read_jsonl(output)) == 1
    class RecordingJudge(FakeJudge):
        async def evaluate(self, prediction):
            calls.append(prediction.case_id)
            return await super().evaluate(prediction)
    calls.clear()
    results = asyncio.run(judge_predictions(predictions=values, output_path=output,
                                           client=RecordingJudge(), resume=True, retry_errors=True))
    assert calls == ["1", "2"] and len(results) == 3
    changed = FakeJudge()
    changed.model = "different-judge"
    with pytest.raises(ValueError, match="model/config/prompt changed"):
        asyncio.run(judge_predictions(predictions=values, output_path=output, client=changed, resume=True))


def test_frozen_benchmark_has_250_original_qa_and_50_behavior_probes():
    root = Path(__file__).parents[1] / "evals/datasets"
    qa = load_cases(root / "qa250_v1/cases.jsonl")
    guards = load_cases(root / "guardrails50_v1/cases.jsonl")
    combined = load_cases(root / "prototype300_v1/cases.jsonl")
    assert (len(qa), len(guards), len(combined)) == (250, 50, 300)
    assert sum(c.source_file == "QA/train.json" for c in qa) == 200
    assert sum(c.source_file == "QA/warmup.json" for c in qa) == 50
    assert all(c.expected_behavior == "answer" for c in qa)
    assert all(c.expected_behavior != "answer" for c in guards)
    assert len({question_key(c.question) for c in combined}) == 300
    old = {question_key(c.question) for folder in ("qa_random30_v0", "qa_random50_v0_2082005")
           for c in load_cases(root / folder / "cases.jsonl")}
    assert not old.intersection(question_key(c.question) for c in qa)
    for folder in ("qa250_v1", "guardrails50_v1", "prototype300_v1"):
        manifest = json.loads((root / folder / "manifest.json").read_text())
        assert manifest["cases_sha256"] == sha256_file(root / folder / "cases.jsonl")


def test_summary_reports_scored_coverage_and_ignores_historical_failures():
    prediction = {"case_id": "a", "prediction": {"abstained": False}, "tags": ["slice:standard"]}
    result = build_summary(requested_case_count=2, predictions=[prediction, prediction],
                           errors=[{"case_id": "a"}], judge_results=[
                               {"case_id": "a", "error": "old"},
                               {"case_id": "a", "assessment": {"overall_pass": True,
                                "correctness": 5, "completeness": 5, "faithfulness": 5,
                                "citation_support": 5, "legal_usefulness": 5},
                                "deterministic": {"passed": True}},
                           ])
    assert result["successful_case_count"] == 1
    assert result["error_case_count"] == 0
    assert result["judge"]["error_count"] == 0
    assert result["judge"]["known_pass_fraction_of_requested"] == 0.5
    assert result["slices"]["slice:standard"]["scored_count"] == 1


def test_sigterm_cli_exits_cleanly_and_releases_run_lock(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    append_jsonl(dataset, _case("1"))
    output = tmp_path / "run"
    script = """
import asyncio, sys
from legal_ir_prototype.evaluation import cli
async def fake_run(**kwargs):
    (kwargs['output_dir'] / 'ready').write_text('ready')
    await asyncio.Event().wait()
cli.run_evaluation = fake_run
sys.exit(cli.main(sys.argv[1:]))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, "run", "--config", "unused.yaml", "--dataset", str(dataset),
         "--output", str(output), "--env-file", str(tmp_path / "none"), "--continue"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not (output / "ready").exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (output / "ready").exists()
        process.terminate()
        _, stderr = process.communicate(timeout=5)
        assert process.returncode == 130
        assert "Re-run the same command with --continue" in stderr
        with artifact_lock(output / ".evaluation.lock"):
            pass
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_all_300_cases_roundtrip_with_fake_generation_and_judge(tmp_path):
    cases = load_cases(Path(__file__).parents[1] / "evals/datasets/prototype300_v1/cases.jsonl")
    recorder = ArtifactRecorder(tmp_path / "run", resume=False)
    asyncio.run(runner.execute_cases(
        rag_service=_rag_service(), cases=cases, recorder=recorder, run_id="fake-validation",
        include_prompt=False, include_raw_llm_response=False,
    ))
    predictions = tuple(runner.EvaluationPrediction.model_validate(p) for p in recorder.predictions())
    results = asyncio.run(judge_predictions(predictions=predictions, output_path=tmp_path / "judge.jsonl",
                                           client=FakeJudge(), resume=False))
    assert len(predictions) == len(results) == 300
    assert {p.expected_behavior for p in predictions} == {"answer", "clarify", "abstain", "safe_response"}
    assert all(r.latency_ms is not None for r in results)
