from __future__ import annotations

"""Aggregate stable evaluation metrics from append-only artifacts."""

import math
from collections import defaultdict
from statistics import fmean
from typing import Any, Iterable, Mapping
from .recorder import latest_by_case


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_summary(
    *,
    requested_case_count: int,
    predictions: Iterable[Mapping[str, Any]],
    errors: Iterable[Mapping[str, Any]],
    judge_results: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    prediction_values = latest_by_case(list(predictions))
    prediction_ids = {str(p["case_id"]) for p in prediction_values}
    error_values = [e for e in latest_by_case(list(errors)) if str(e["case_id"]) not in prediction_ids]
    judge_values = [j for j in latest_by_case(list(judge_results)) if str(j["case_id"]) in prediction_ids]
    completed = len(prediction_values) + len(error_values)
    abstained = sum(
        bool(value.get("prediction", {}).get("abstained"))
        for value in prediction_values
    )
    timings: dict[str, list[float]] = defaultdict(list)
    for value in prediction_values:
        for stage, duration in value.get("timings_ms", {}).items():
            timings[str(stage)].append(float(duration))
    latency = {
        stage: {
            "mean": round(fmean(values), 3),
            "p50": round(_percentile(values, 0.50) or 0.0, 3),
            "p95": round(_percentile(values, 0.95) or 0.0, 3),
            "max": round(max(values), 3),
        }
        for stage, values in sorted(timings.items())
    }
    input_tokens = [
        value["prediction"].get("input_tokens")
        for value in prediction_values
        if value.get("prediction", {}).get("input_tokens") is not None
    ]
    output_tokens = [
        value["prediction"].get("output_tokens")
        for value in prediction_values
        if value.get("prediction", {}).get("output_tokens") is not None
    ]
    assessments = [
        value["assessment"]
        for value in judge_values
        if isinstance(value.get("assessment"), dict)
    ]
    score_names = (
        "correctness",
        "completeness",
        "faithfulness",
        "citation_support",
        "legal_usefulness",
    )
    judge_summary: dict[str, Any] | None = None
    if judge_values:
        by_case = {str(value.get("case_id")): value for value in judge_values}
        combined_passes = [
            bool(value.get("deterministic", {}).get("passed"))
            and bool(value.get("assessment", {}).get("overall_pass"))
            for value in by_case.values()
            if isinstance(value.get("assessment"), dict)
        ]
        judge_input_tokens = [
            int(value["input_tokens"])
            for value in judge_values
            if value.get("input_tokens") is not None
        ]
        judge_output_tokens = [
            int(value["output_tokens"])
            for value in judge_values
            if value.get("output_tokens") is not None
        ]
        judge_summary = {
            "result_count": len(judge_values),
            "scored_count": len(assessments),
            "scoring_coverage": len(assessments) / len(prediction_values) if prediction_values else 0.0,
            "unscored_prediction_count": len(prediction_values) - len(assessments),
            "pass_rate_denominator": "successfully_scored_cases_only",
            "known_pass_fraction_of_requested": sum(combined_passes) / requested_case_count if requested_case_count else 0.0,
            "error_count": sum(bool(value.get("error")) for value in judge_values),
            "deterministic_pass_rate": (
                sum(bool(value.get("deterministic", {}).get("passed")) for value in judge_values)
                / len(judge_values)
            ),
            "llm_judge_pass_rate": (
                None
                if not assessments
                else sum(bool(value.get("overall_pass")) for value in assessments)
                / len(assessments)
            ),
            "overall_pass_rate": (
                None
                if not combined_passes
                else sum(combined_passes) / len(combined_passes)
            ),
            "tokens": {
                "input_total": sum(judge_input_tokens),
                "output_total": sum(judge_output_tokens),
            },
            "mean_scores": {
                name: (
                    None
                    if not assessments
                    else round(fmean(float(value[name]) for value in assessments), 4)
                )
                for name in score_names
            },
        }
    result = {
        "schema_version": 1,
        "requested_case_count": requested_case_count,
        "completed_case_count": completed,
        "successful_case_count": len(prediction_values),
        "error_case_count": len(error_values),
        "completion_rate": completed / requested_case_count if requested_case_count else 0.0,
        "error_rate": len(error_values) / requested_case_count if requested_case_count else 0.0,
        "abstention_count": abstained,
        "abstention_rate": abstained / len(prediction_values) if prediction_values else 0.0,
        "tokens": {
            "input_total": sum(input_tokens),
            "output_total": sum(output_tokens),
            "input_mean": None if not input_tokens else round(fmean(input_tokens), 3),
            "output_mean": None if not output_tokens else round(fmean(output_tokens), 3),
        },
        "latency_ms": latency,
        "judge": judge_summary,
    }
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    judges = {str(j["case_id"]): j for j in judge_values}
    for value in prediction_values:
        for tag in value.get("tags", []):
            if str(tag).startswith("slice:"):
                groups[str(tag)].append(value)
    result["slices"] = {}
    for tag, values in sorted(groups.items()):
        scored = [judges[str(v["case_id"])] for v in values
                  if str(v["case_id"]) in judges and isinstance(judges[str(v["case_id"])].get("assessment"), dict)]
        result["slices"][tag] = {
            "successful_case_count": len(values), "scored_count": len(scored),
            "abstention_count": sum(bool(v["prediction"]["abstained"]) for v in values),
            "judge_pass_rate_on_scored": (sum(bool(j["assessment"]["overall_pass"]) for j in scored) / len(scored)) if scored else None,
        }
    return result
