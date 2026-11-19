from __future__ import annotations

"""Command line interface for frozen-dataset RAG evaluation."""

import argparse
import asyncio
import json
import logging
import math
import signal
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from ..core import AppConfig
from ..environment import load_environment_file
from .dataset import load_cases, prepare_dataset, sha256_file, write_dataset
from .judge import OpenAICompatibleJudge, judge_predictions
from .recorder import artifact_lock, atomic_write_json, latest_by_case, read_jsonl
from .runner import run_evaluation
from .schemas import EvaluationPrediction
from .summary import build_summary
from .trace import EvaluationTrace, Pricing
from .guardrails import prepare_guardrails
from ..llm.transport import RetryPolicy
from .analysis import write_analysis
from .diagnostic_judge import DiagnosticJudge, judge_diagnostic_predictions, request_usage_summary


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _non_negative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _provider_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--continue", dest="continue_run", action="store_true",
                        help="Resume, skip successful cases, and retry failed/pending cases.")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--request-timeout", type=_positive_float, default=90.0)
    parser.add_argument("--retry-base-delay", type=_non_negative_float, default=2.0)
    parser.add_argument("--retry-max-delay", type=_non_negative_float, default=60.0)
    parser.add_argument("--retry-budget", type=_positive_float, default=300.0)
    parser.add_argument("--request-interval", type=_non_negative_float, default=5.0,
                        help="Minimum seconds between HTTP attempt starts, including retries.")
    parser.add_argument("--input-usd-per-million", type=_non_negative_float)
    parser.add_argument("--output-usd-per-million", type=_non_negative_float)


def _retry_policy(args: argparse.Namespace) -> RetryPolicy:
    return RetryPolicy(max_retries=args.max_retries,
                       base_delay_seconds=args.retry_base_delay,
                       max_delay_seconds=args.retry_max_delay,
                       budget_seconds=args.retry_budget,
                       request_interval_seconds=args.request_interval)


def _pricing(args: argparse.Namespace) -> Pricing:
    return Pricing(args.input_usd_per_million, args.output_usd_per_million)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-rag-eval",
        description="Prepare, run, and judge reproducible legal RAG evaluations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Freeze a seeded QA sample.")
    prepare.add_argument("--source", type=Path, action="append", required=True)
    prepare.add_argument("--sample-size", type=int, default=30)
    prepare.add_argument("--seed", type=int, default=2026)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument("--strategy", choices=("random", "stratified"), default="random")
    prepare.add_argument("--source-counts", type=int, nargs="+")
    prepare.add_argument("--exclude-dataset", type=Path, action="append", default=[])

    combine = subparsers.add_parser("combine", help="Combine frozen datasets without changing answers.")
    combine.add_argument("--dataset", type=Path, action="append", required=True)
    combine.add_argument("--output", type=Path, required=True)
    guards = subparsers.add_parser("prepare-guardrails", help="Freeze 50 behavioral guardrail probes.")
    guards.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run", help="Run the online RAG pipeline offline.")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--env-file", type=Path, default=Path(".env"))
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--limit",
        type=int,
        help="Run only the first N frozen cases (use a separate smoke output).",
    )
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--retry-errors",
        action="store_true",
        help="With --resume, retry only cases whose latest run attempt failed.",
    )
    run.add_argument("--include-prompt", action="store_true")
    run.add_argument("--include-raw-llm-response", action="store_true")
    run.add_argument("--require-cuda", action="store_true")
    _provider_arguments(run)

    judge = subparsers.add_parser("judge", help="Run deterministic and LLM judging.")
    judge.add_argument("--predictions", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--config", type=Path)
    judge.add_argument("--env-file", type=Path, default=Path(".env"))
    judge.add_argument("--resume", action="store_true")
    judge.add_argument(
        "--retry-errors",
        action="store_true",
        help="With --resume, retry only cases whose latest judge attempt failed.",
    )
    _provider_arguments(judge)

    diagnostic = subparsers.add_parser("judge-diagnostic", help="Judge multiple samples in one prompt and classify evidence/behavior outcomes.")
    diagnostic.add_argument("--predictions", type=Path, required=True)
    diagnostic.add_argument("--output", type=Path, required=True)
    diagnostic.add_argument("--config", type=Path)
    diagnostic.add_argument("--env-file", type=Path, default=Path(".env"))
    diagnostic.add_argument("--resume", action="store_true")
    diagnostic.add_argument("--retry-errors", action="store_true")
    diagnostic.add_argument("--samples-per-prompt", type=_positive_int, default=5)
    diagnostic.add_argument("--max-prompt-characters", type=_positive_int, default=300_000)
    diagnostic.add_argument("--output-tokens-per-sample", type=_positive_int, default=2048)
    _provider_arguments(diagnostic)

    analysis = subparsers.add_parser("analyze", help="Classify saved diagnostic judgments and compare legal references without an API call.")
    analysis.add_argument("--predictions", type=Path, required=True)
    analysis.add_argument("--judge-results", type=Path)
    analysis.add_argument("--output", type=Path, required=True)
    return parser


def _config_from_manifest(predictions_path: Path) -> Path:
    manifest_path = predictions_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "--config is required when run_manifest.json is not next to predictions"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = manifest.get("runtime_snapshot") or manifest.get("runtime_config")
    if not config_path:
        raise ValueError("run manifest does not contain runtime_config")
    path = Path(str(config_path))
    expected = manifest.get("runtime_snapshot_sha256") if manifest.get("runtime_snapshot") else manifest.get("config_sha256")
    if expected and sha256_file(path) != expected:
        raise ValueError("recorded runtime config changed; supply an explicit --config for a new judge output")
    return path


async def _judge(args: argparse.Namespace) -> dict[str, object]:
    load_environment_file(args.env_file)
    config_path = args.config or _config_from_manifest(args.predictions)
    config = AppConfig.from_yaml(config_path)
    if config.llm is None:
        raise ValueError("judge requires an llm configuration")
    predictions = tuple(
        EvaluationPrediction.model_validate(value)
        for value in latest_by_case(read_jsonl(args.predictions))
    )
    trace = EvaluationTrace(args.output.with_suffix(".trace.jsonl"), stage="judge",
                            pricing=_pricing(args), resume=args.resume)
    client = OpenAICompatibleJudge(replace(config.llm, timeout_seconds=args.request_timeout),
                                  retry_policy=_retry_policy(args), attempt_observer=trace.observe)
    run_dir = args.predictions.parent
    prediction_values = latest_by_case(read_jsonl(args.predictions))
    prediction_ids = {str(value["case_id"]) for value in prediction_values}
    errors = [
        value
        for value in latest_by_case(read_jsonl(run_dir / "errors.jsonl"))
        if str(value["case_id"]) not in prediction_ids
    ]
    manifest_path = run_dir / "run_manifest.json"
    requested = len(predictions) + len(errors)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        requested = int(manifest.get("requested_case_count", requested))
    def checkpoint() -> dict[str, object]:
        summary = build_summary(requested_case_count=requested,
                                predictions=prediction_values,
                                errors=errors, judge_results=read_jsonl(args.output))
        summary["judge_provider_trace"] = trace.summary()
        summary["judge_results_path"] = str(args.output.resolve())
        generation_trace = run_dir / "generation_trace.jsonl"
        if generation_trace.exists():
            summary["provider_trace"] = EvaluationTrace(generation_trace, stage="generation",
                                                        pricing=Pricing(), resume=True).summary()
        atomic_write_json(args.output.with_suffix(".summary.json"), summary)
        atomic_write_json(run_dir / "summary.json", summary)
        return summary
    try:
        await judge_predictions(predictions=predictions, output_path=args.output,
                                client=client, resume=args.resume,
                                retry_errors=args.retry_errors, on_progress=checkpoint)
    finally:
        summary = checkpoint()
        await client.aclose()
    return summary


async def _judge_diagnostic(args: argparse.Namespace) -> dict[str, object]:
    load_environment_file(args.env_file)
    config = AppConfig.from_yaml(args.config or _config_from_manifest(args.predictions))
    if config.llm is None:
        raise ValueError("diagnostic judge requires an llm configuration")
    if not args.predictions.is_file():
        raise FileNotFoundError(args.predictions)
    values = latest_by_case(read_jsonl(args.predictions))
    predictions = tuple(EvaluationPrediction.model_validate(v) for v in values)
    if not predictions:
        raise ValueError("no predictions to judge")
    trace = EvaluationTrace(args.output.with_suffix(".trace.jsonl"), stage="diagnostic_judge",
                            pricing=_pricing(args), resume=args.resume)
    client = DiagnosticJudge(replace(config.llm, timeout_seconds=args.request_timeout),
                             retry_policy=_retry_policy(args), attempt_observer=trace.observe,
                             output_tokens_per_sample=args.output_tokens_per_sample)
    manifest_path = args.predictions.parent / "run_manifest.json"
    requested = len(predictions)
    if manifest_path.exists():
        requested = int(json.loads(manifest_path.read_text()).get("requested_case_count", requested))
    errors = [e for e in latest_by_case(read_jsonl(args.predictions.parent / "errors.jsonl"))
              if str(e["case_id"]) not in {p.case_id for p in predictions}]
    def checkpoint() -> dict[str, object]:
        all_judgments = read_jsonl(args.output)
        judgments = latest_by_case(all_judgments)
        summary = build_summary(requested_case_count=requested, predictions=values,
                                errors=errors, judge_results=judgments)
        usage = request_usage_summary(all_judgments)
        summary["judge_request_usage"] = usage
        if summary["judge"] is not None:
            summary["judge"]["tokens"] = {"input_total": usage["input_total"], "output_total": usage["output_total"]}
        summary["judge_provider_trace"] = trace.summary()
        summary["judge_results_path"] = str(args.output.resolve())
        atomic_write_json(args.output.with_suffix(".summary.json"), summary)
        return summary
    try:
        await judge_diagnostic_predictions(predictions=predictions, output_path=args.output, client=client,
                                            resume=args.resume, retry_errors=args.retry_errors,
                                            samples_per_prompt=args.samples_per_prompt,
                                            max_prompt_characters=args.max_prompt_characters,
                                            on_progress=checkpoint)
    finally:
        try:
            summary = checkpoint()
            analysis_dir = args.output.parent / f"{args.output.stem}.analysis"
            if args.output.exists():
                analysis_summary = write_analysis(args.predictions, args.output, analysis_dir)
                summary["analysis_directory"] = str(analysis_dir.resolve())
                summary["diagnostic_outcomes"] = analysis_summary["outcomes"]
                atomic_write_json(args.output.with_suffix(".summary.json"), summary)
        finally:
            await client.aclose()
    return summary


async def _interruptible(coroutine):
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await coroutine
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if getattr(args, "continue_run", False):
        args.resume = args.retry_errors = True
    if getattr(args, "retry_errors", False) and not getattr(args, "resume", False):
        parser.error("--retry-errors requires --resume")
    if args.command == "prepare":
        manifest = prepare_dataset(
            source=args.source,
            output_dir=args.output,
            sample_size=args.sample_size,
            seed=args.seed,
            overwrite=args.overwrite,
            strategy=args.strategy, source_counts=args.source_counts,
            exclude_datasets=args.exclude_dataset,
        )
        print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0
    if args.command == "combine":
        cases = tuple(c for path in args.dataset for c in load_cases(path))
        manifest = write_dataset(cases=cases, output_dir=args.output,
                                 selection_method="combined_suite",
                                 sources=[{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in args.dataset])
        print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare-guardrails":
        manifest = prepare_guardrails(args.output)
        print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        load_environment_file(args.env_file)
        cases = load_cases(args.dataset)
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("--limit must be positive")
            cases = cases[: args.limit]
        _retry_policy(args)
        _pricing(args)
        with artifact_lock(args.output / ".evaluation.lock"):
            summary = asyncio.run(_interruptible(run_evaluation(
                config_path=args.config,
                dataset_path=args.dataset,
                cases=cases,
                output_dir=args.output,
                resume=args.resume,
                retry_errors=args.retry_errors,
                include_prompt=args.include_prompt,
                include_raw_llm_response=args.include_raw_llm_response,
                require_cuda=args.require_cuda,
                retry_policy=_retry_policy(args), request_timeout_seconds=args.request_timeout,
                pricing=_pricing(args),
            )))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "judge":
        _retry_policy(args)
        _pricing(args)
        with artifact_lock(args.predictions.parent / ".evaluation.lock"), artifact_lock(args.output.with_suffix(".lock")):
            summary = asyncio.run(_interruptible(_judge(args)))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "judge-diagnostic":
        _retry_policy(args)
        _pricing(args)
        with artifact_lock(args.predictions.parent / ".evaluation.lock"), artifact_lock(args.output.with_suffix(".lock")):
            summary = asyncio.run(_interruptible(_judge_diagnostic(args)))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "analyze":
        with artifact_lock(args.output / ".analysis.lock"):
            summary = write_analysis(args.predictions, args.judge_results, args.output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("Evaluation interrupted; saved results are retained. Re-run the same command with --continue.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
