from __future__ import annotations

"""Deterministic QA checks that do not require another model call."""

from .schemas import DeterministicEvaluation, EvaluationPrediction


def run_deterministic_checks(
    prediction: EvaluationPrediction,
) -> DeterministicEvaluation:
    response = prediction.prediction
    context_source_ids = {
        str(value.get("source_id"))
        for value in prediction.context.get("sources", [])
        if isinstance(value, dict) and value.get("source_id")
    }
    citation_ids = {
        str(value.get("source_id"))
        for value in response.citations
        if isinstance(value, dict) and value.get("source_id")
    }
    checks = {
        "output_schema_valid": True,
        "answer_non_empty": bool(response.answer.strip()),
        "citation_ids_exist_in_context": citation_ids <= context_source_ids,
        "answered_response_has_citation": response.abstained or bool(citation_ids),
        "abstention_has_no_citations": not response.abstained or not citation_ids,
        "abstention_has_reason": (
            not response.abstained or bool((response.abstention_reason or "").strip())
        ),
        "non_abstention_has_no_reason": (
            response.abstained or response.abstention_reason is None
        ),
    }
    errors = tuple(name for name, passed in checks.items() if not passed)
    return DeterministicEvaluation(
        passed=not errors,
        checks=checks,
        errors=errors,
    )
