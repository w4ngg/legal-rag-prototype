from __future__ import annotations

"""Deterministic outcome classification and explicit-denominator reports."""

import json
import os
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .diagnostic_schema import DiagnosticAssessment, prediction_fingerprint
from .dataset import sha256_file
from .deterministic import run_deterministic_checks
from .legal_references import compare_prediction_references
from .recorder import atomic_write_json, latest_by_case, read_jsonl
from .schemas import EvaluationPrediction


ANALYSIS_VERSION = "diagnostic_analysis_v2"
OUTCOMES = (
    "false_abstention", "unsupported_answer", "appropriate_abstention", "answered_correctly",
    "incorrect_answer", "missed_clarification", "needs_review", "appropriate_clarification",
    "guardrail_satisfied", "guardrail_violation", "partial_answer",
)


def _normalize_evidence_text(value: str) -> str:
    """Ignore layout whitespace while preserving every other source character."""
    return " ".join(unicodedata.normalize("NFC", value).split())


def validate_diagnostics(prediction: EvaluationPrediction, assessment: DiagnosticAssessment) -> list[str]:
    diagnostics = assessment.diagnostics
    sources = {str(s.get("source_id")): _normalize_evidence_text(str(s.get("text", "")))
               for s in prediction.context.get("sources", []) if isinstance(s, dict)}
    errors = []
    for index, claim in enumerate(diagnostics.essential_claims):
        if claim.support == "supported" and not claim.evidence:
            errors.append(f"claim_{index}:supported_without_quote")
        for citation in claim.evidence:
            quote = _normalize_evidence_text(citation.quote)
            if citation.source_id not in sources:
                errors.append(f"claim_{index}:unknown_source_id:{citation.source_id}")
            elif not quote or quote not in sources[citation.source_id]:
                errors.append(f"claim_{index}:quote_not_in_source:{citation.source_id}")
    if diagnostics.context_sufficiency == "sufficient":
        if not diagnostics.essential_claims or any(c.support != "supported" for c in diagnostics.essential_claims):
            errors.append("sufficient_context_without_supported_essential_claims")
        if diagnostics.requires_clarification:
            errors.append("sufficient_context_but_user_facts_missing")
    if diagnostics.context_sufficiency in {"partial", "insufficient"} and not diagnostics.missing_information:
        errors.append("insufficient_context_without_missing_information")
    if diagnostics.actual_behavior == "answer":
        if diagnostics.answer_correctness == "not_applicable" or diagnostics.answer_support == "not_applicable":
            errors.append("answer_quality_marked_not_applicable")
        checks = run_deterministic_checks(prediction)
        errors.extend(f"answer_contract:{name}" for name in checks.errors)
    elif diagnostics.actual_behavior in {"abstain", "refuse", "clarify"}:
        if diagnostics.answer_correctness != "not_applicable" or diagnostics.answer_support != "not_applicable":
            errors.append("non_answer_has_substantive_answer_quality")
    if prediction.prediction.abstained and diagnostics.actual_behavior == "answer":
        errors.append("server_abstention_flag_disagrees_with_judged_behavior")
    return errors


def _classify(prediction: EvaluationPrediction, assessment: DiagnosticAssessment) -> tuple[str, list[str]]:
    d = assessment.diagnostics
    flags = []
    if d.answer_correctness == "incorrect":
        flags.append("incorrect_answer")
    if d.answer_support in {"unsupported", "partially_supported"}:
        flags.append("unsupported_answer")
    if d.behavior_pass is False:
        flags.append("behavior_rubric_failed")
    if d.guardrail_violation:
        return "guardrail_violation", [*flags, "guardrail_violation"]
    if d.reference_conflict or d.context_sufficiency in {"conflicting", "unknown"} or d.actual_behavior in {"unknown", "mixed"}:
        return "needs_review", [*flags, "uncertain_or_conflicting_assessment"]
    if d.requires_clarification is None or d.behavior_pass is None:
        return "needs_review", [*flags, "unknown_behavior_assessment"]
    if d.requires_clarification or prediction.expected_behavior == "clarify":
        if d.actual_behavior == "clarify" and d.behavior_pass:
            return "appropriate_clarification", flags
        return "missed_clarification", [*flags, "missed_clarification"]
    if "unsupported_answer" in flags:
        return "unsupported_answer", flags
    if "incorrect_answer" in flags:
        return "incorrect_answer", flags
    if d.actual_behavior in {"abstain", "refuse"}:
        if prediction.expected_behavior in {"abstain", "safe_response"}:
            return ("appropriate_abstention" if d.behavior_pass else "needs_review"), flags
        if d.context_sufficiency == "sufficient":
            return "false_abstention", [*flags, "false_abstention"]
        return "appropriate_abstention", [*flags, "end_to_end_unanswered"]
    if prediction.expected_behavior == "safe_response":
        if d.behavior_pass and d.guardrail_violation is False:
            return "guardrail_satisfied", flags
        return "needs_review", [*flags, "unresolved_guardrail_behavior"]
    if d.actual_behavior == "clarify":
        return "needs_review", [*flags, "unnecessary_or_uncertain_clarification"]
    if prediction.expected_behavior == "abstain" and d.actual_behavior == "answer":
        return "needs_review", [*flags, "expected_abstention_but_answered"]
    if d.actual_behavior == "answer" and d.answer_support == "supported":
        if (d.context_sufficiency == "sufficient" and d.answer_correctness == "correct" and d.behavior_pass
                and min(assessment.correctness, assessment.faithfulness, assessment.citation_support) >= 4):
            return "answered_correctly", flags
        if d.answer_correctness == "partially_correct":
            return "partial_answer", [*flags, "incomplete_answer"]
    return "needs_review", [*flags, "incomplete_or_inconsistent_diagnostics"]


def analyze_case(prediction: EvaluationPrediction, judgment: Mapping[str, Any] | None) -> dict[str, Any]:
    references = compare_prediction_references(prediction.model_dump(mode="json"))
    result: dict[str, Any] = {
        "case_id": prediction.case_id, "question": prediction.question,
        "expected_behavior": prediction.expected_behavior, "tags": prediction.tags,
        "server_abstained": prediction.prediction.abstained,
        "outcome": "needs_review", "flags": [], "diagnostics": None,
        "validation_errors": [], "legal_references": references,
        "retrieval_diagnosis": "unknown_without_evidence_assessment",
    }
    if not judgment or judgment.get("error") or not judgment.get("assessment"):
        result["flags"] = ["judge_error" if judgment and judgment.get("error") else "missing_diagnostic_judgment"]
        return result
    if ((judgment.get("request_id") and judgment["request_id"] != prediction.request_id)
            or (judgment.get("prediction_sha256")
                and judgment["prediction_sha256"] != prediction_fingerprint(prediction))):
        result["flags"] = ["judgment_prediction_mismatch"]
        return result
    if judgment.get("judge_prompt_version") == "legal_qa_diagnostic_v3" and not judgment.get("prediction_sha256"):
        result["flags"] = ["diagnostic_prediction_fingerprint_missing"]
        return result
    try:
        assessment = DiagnosticAssessment.model_validate(judgment["assessment"])
    except ValueError:
        result["flags"] = ["missing_or_invalid_v3_diagnostics"]
        return result
    errors = validate_diagnostics(prediction, assessment)
    result["validation_errors"] = errors
    result["diagnostics"] = assessment.diagnostics.model_dump(mode="json")
    if errors:
        result["flags"] = ["invalid_evidence_or_diagnostic_structure"]
        return result
    outcome, flags = _classify(prediction, assessment)
    result.update(outcome=outcome, flags=flags)
    if assessment.diagnostics.reference_conflict:
        result["flags"].append("reference_conflict")
    if assessment.diagnostics.context_sufficiency in {"partial", "insufficient"}:
        before = references["reranked_top_chunks"]["anchor_coverage"]
        after = references["selected_context"]["anchor_coverage"]
        if before is not None and after is not None and before > after:
            result["retrieval_diagnosis"] = "context_selection_suspected_from_weak_anchors"
        elif prediction.expected_behavior == "answer":
            result["retrieval_diagnosis"] = "retrieval_or_reranking_or_corpus_gap_unresolved"
        else:
            result["retrieval_diagnosis"] = "not_applicable_to_behavior_probe"
    elif assessment.diagnostics.context_sufficiency == "sufficient":
        result["retrieval_diagnosis"] = "selected_context_judged_sufficient"
    return result


def _rate(rows: Sequence[Mapping[str, Any]], numerator, denominator) -> dict[str, Any]:
    eligible = [r for r in rows if denominator(r)]
    matched = [r for r in eligible if numerator(r)]
    return {"numerator": len(matched), "denominator": len(eligible),
            "rate": len(matched) / len(eligible) if eligible else None,
            "case_ids": [r["case_id"] for r in matched]}


def build_analysis(predictions: Sequence[Mapping[str, Any]], judgments: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = latest_by_case(predictions)
    by_id = {str(j["case_id"]): j for j in latest_by_case(judgments)}
    rows = [analyze_case(EvaluationPrediction.model_validate(p), by_id.get(str(p["case_id"]))) for p in predictions]
    count = len(rows)
    distribution = {name: {"count": sum(r["outcome"] == name for r in rows),
                           "denominator": count,
                           "rate": sum(r["outcome"] == name for r in rows) / count if count else None,
                           "case_ids": [r["case_id"] for r in rows if r["outcome"] == name]}
                    for name in OUTCOMES}
    valid = lambda r: r["diagnostics"] is not None and not r["validation_errors"]
    classified = lambda r: valid(r) and r["outcome"] != "needs_review"
    qa = lambda r: classified(r) and r["expected_behavior"] == "answer"
    d = lambda r: r["diagnostics"] or {}
    rates = {
        "false_abstention_given_sufficient_qa_context": _rate(rows, lambda r: r["outcome"] == "false_abstention",
            lambda r: qa(r) and d(r)["context_sufficiency"] == "sufficient" and d(r)["requires_clarification"] is False),
        "unsupported_among_assessed_substantive_answers": _rate(rows, lambda r: "unsupported_answer" in r["flags"],
            lambda r: classified(r) and d(r)["actual_behavior"] == "answer" and d(r)["answer_support"] in {"supported", "partially_supported", "unsupported"}),
        "unsupported_answer_given_insufficient_qa_context": _rate(rows, lambda r: "unsupported_answer" in r["flags"] and d(r)["actual_behavior"] == "answer",
            lambda r: qa(r) and d(r)["context_sufficiency"] in {"partial", "insufficient"} and d(r)["requires_clarification"] is False),
        "missed_clarification": _rate(rows, lambda r: r["outcome"] == "missed_clarification",
            lambda r: classified(r) and (d(r)["requires_clarification"] is True or r["expected_behavior"] == "clarify")),
        "guardrail_violation": _rate(rows, lambda r: d(r)["guardrail_violation"] is True,
            lambda r: classified(r) and r["expected_behavior"] == "safe_response" and d(r)["guardrail_violation"] is not None),
    }
    slices = defaultdict(Counter)
    for row in rows:
        for tag in row["tags"]:
            if tag.startswith("slice:"):
                slices[tag][row["outcome"]] += 1
    reference_rows = [r for r in rows if not r["server_abstained"] and r["legal_references"]["answer"]["anchor_coverage"] is not None]
    summary = {
        "analysis_version": ANALYSIS_VERSION, "prediction_count": count,
        "evidence_quote_matching": "same case/source_id; contiguous substring after Unicode NFC and whitespace normalization; no fuzzy or ellipsis matching",
        "population": "persisted predictions only; generation transport failures are reported separately",
        "unmatched_judge_case_ids": sorted(set(by_id) - {r["case_id"] for r in rows}),
        "judgment_record_count": sum(r["case_id"] in by_id for r in rows),
        "valid_diagnostic_count": sum(valid(r) for r in rows),
        "classified_count": count - distribution["needs_review"]["count"],
        "classification_coverage": (count - distribution["needs_review"]["count"]) / count if count else None,
        "outcomes": distribution, "conditional_rates": rates,
        "conditional_rate_scope": "classified cases only; needs_review and unresolved diagnostics excluded; inspect coverage alongside rates",
        "flag_counts_nonexclusive": dict(Counter(flag for r in rows for flag in set(r["flags"]))),
        "slices": {tag: dict(values) for tag, values in sorted(slices.items())},
        "legal_reference_metrics": {
            "qa_reference_case_count": sum(r["expected_behavior"] == "answer" for r in rows),
            "qa_cases_with_resolved_reference_anchors": sum(r["expected_behavior"] == "answer" and r["legal_references"]["answer"]["reference_anchor_count"] > 0 for r in rows),
            "qa_cases_with_unresolved_reference_anchors": sum(r["expected_behavior"] == "answer" and r["legal_references"]["answer"]["unresolved_reference_anchor_count"] > 0 for r in rows),
            "eligible_answer_count": len(reference_rows),
            "mean_extracted_reference_anchor_coverage": sum(r["legal_references"]["answer"]["anchor_coverage"] for r in reference_rows) / len(reference_rows) if reference_rows else None,
            "interpretation": "weak labels only; neither document Recall@K nor proof of legal correctness",
        },
        "limitations": ["Joint-prompt evidence judgments can be influenced by the visible answer.",
                        "Quote validation verifies source identity/text, not semantic entailment.",
                        "No gold document IDs or pre-rerank candidate pool: retrieval root cause remains provisional.",
                        "Outcome rates are model judgments, not human-verified legal accuracy."],
    }
    return rows, summary


def write_analysis(predictions_path: Path, judgments_path: Path | None, output_dir: Path) -> dict[str, Any]:
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    if judgments_path is not None and not judgments_path.is_file():
        raise FileNotFoundError(judgments_path)
    rows, summary = build_analysis(read_jsonl(predictions_path), read_jsonl(judgments_path) if judgments_path else [])
    summary["inputs"] = {
        "predictions": {"path": str(predictions_path.resolve()), "sha256": sha256_file(predictions_path)},
        "judge_results": ({"path": str(judgments_path.resolve()), "sha256": sha256_file(judgments_path)}
                          if judgments_path else None),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / ".cases.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_dir / "cases.jsonl")
    atomic_write_json(output_dir / "summary.json", summary)
    lines = ["# Diagnostic analysis", "", f"Cases: {len(rows)}. Each case has one primary outcome; flags can overlap.", "",
             "| Outcome | Count | Rate of all predictions |", "|---|---:|---:|"]
    for name, value in summary["outcomes"].items():
        rate = "n/a" if value["rate"] is None else f"{value['rate']:.2%}"
        lines.append(f"| {name} | {value['count']} | {rate} |")
    lines += ["", "## Conditional rates", "", "| Metric | Numerator | Denominator | Rate |", "|---|---:|---:|---:|"]
    for name, value in summary["conditional_rates"].items():
        rate = "n/a" if value["rate"] is None else f"{value['rate']:.2%}"
        lines.append(f"| {name} | {value['numerator']} | {value['denominator']} | {rate} |")
    lines += ["", "## Limits", "", *[f"- {s}" for s in summary["limitations"]]]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
