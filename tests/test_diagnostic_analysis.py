from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from legal_ir_prototype.evaluation.analysis import analyze_case, build_analysis, write_analysis
from legal_ir_prototype.evaluation.diagnostic_schema import DiagnosticAssessment, prediction_fingerprint
from legal_ir_prototype.evaluation.recorder import append_jsonl
from legal_ir_prototype.evaluation.schemas import EvaluationPrediction


SOURCE = "Điều 5. Người lao động được nghỉ hằng năm theo các điều kiện được quy định."


def prediction(case_id="case-1", *, expected="answer", abstained=False):
    return EvaluationPrediction.model_validate({
        "run_id": "r", "request_id": f"req-{case_id}", "case_id": case_id,
        "source_query_id": case_id, "question": "Người lao động được nghỉ thế nào?",
        "reference_answer": "Theo Điều 5 Nghị định 12/2022/NĐ-CP, người lao động được nghỉ hằng năm.",
        "prediction": {"answer": "Không đủ thông tin." if abstained else "Theo Điều 5 Nghị định 12/2022/NĐ-CP, người lao động được nghỉ hằng năm.",
                       "citations": [] if abstained else [{"source_id": "SOURCE_1", "document_id": "doc1"}],
                       "abstained": abstained, "abstention_reason": "insufficient" if abstained else None,
                       "warnings": [], "model": "fake"},
        "retrieved_chunks": [{"document_id": "doc1", "passage": SOURCE,
                              "metadata": {"document_name": "Nghi-dinh-12-2022-ND-CP"}}],
        "context": {"sources": [{"source_id": "SOURCE_1", "document_id": "doc1", "text": SOURCE,
                                 "metadata": {"document_name": "Nghi-dinh-12-2022-ND-CP"}}]},
        "timings_ms": {}, "expected_behavior": expected,
        "tags": ["slice:standard" if expected == "answer" else "slice:guardrails"],
    })


def assessment(*, context="sufficient", behavior="answer", correct="correct", support="supported",
               clarify=False, behavior_pass=True, violation=False, conflict=False):
    return DiagnosticAssessment.model_validate({
        "correctness": 5, "completeness": 5, "faithfulness": 5, "citation_support": 5,
        "legal_usefulness": 5, "overall_pass": True, "rationale": "Đánh giá minh họa.",
        "diagnostics": {
            "context_sufficiency": context, "actual_behavior": behavior,
            "answer_correctness": correct, "answer_support": support,
            "requires_clarification": clarify, "behavior_pass": behavior_pass,
            "guardrail_violation": violation, "reference_conflict": conflict,
            "legal_basis_relation": "match",
            "essential_claims": [{"claim": "Có quyền nghỉ hằng năm", "support": "supported",
                                  "evidence": [{"source_id": "SOURCE_1", "quote": SOURCE}]}],
            "missing_information": [] if context == "sufficient" else ["Thiếu điều kiện áp dụng."],
        },
    })


def judgment(case_id, value):
    return {"case_id": case_id, "assessment": value.model_dump(mode="json"), "error": None}


@pytest.mark.parametrize("kwargs,expected,abstained,outcome", [
    ({}, "answer", False, "answered_correctly"),
    ({"behavior": "abstain", "correct": "not_applicable", "support": "not_applicable"}, "answer", True, "false_abstention"),
    ({"context": "insufficient", "support": "unsupported", "correct": "incorrect"}, "answer", False, "unsupported_answer"),
    ({"context": "insufficient", "behavior": "abstain", "correct": "not_applicable", "support": "not_applicable"}, "answer", True, "appropriate_abstention"),
    ({"correct": "incorrect"}, "answer", False, "incorrect_answer"),
    ({"context": "insufficient", "clarify": True, "behavior": "abstain", "correct": "not_applicable", "support": "not_applicable"}, "clarify", True, "missed_clarification"),
    ({"context": "insufficient", "clarify": True, "behavior": "clarify", "correct": "not_applicable", "support": "not_applicable"}, "clarify", False, "appropriate_clarification"),
    ({"context": "unknown"}, "answer", False, "needs_review"),
    ({"conflict": True}, "answer", False, "needs_review"),
    ({"violation": True}, "safe_response", False, "guardrail_violation"),
    ({}, "safe_response", False, "guardrail_satisfied"),
    ({"context": "partial", "correct": "partially_correct"}, "answer", False, "partial_answer"),
])
def test_outcomes_are_determined_by_code(kwargs, expected, abstained, outcome):
    p = prediction(expected=expected, abstained=abstained)
    a = assessment(**kwargs)
    row = analyze_case(p, judgment(p.case_id, a))
    assert row["outcome"] == outcome


def test_reference_match_never_overrides_unsupported_content_or_false_abstention():
    p = prediction()
    row = analyze_case(p, judgment(p.case_id, assessment(context="insufficient", support="unsupported")))
    assert row["legal_references"]["answer"]["anchor_coverage"] == 1
    assert row["outcome"] == "unsupported_answer"


def test_quote_from_other_case_is_rejected_even_with_same_source_id():
    p = prediction(abstained=True)
    a = assessment(behavior="abstain", correct="not_applicable", support="not_applicable")
    a.diagnostics.essential_claims[0].evidence[0].quote = "Một đoạn của case khác."
    row = analyze_case(p, judgment(p.case_id, a))
    assert row["outcome"] == "needs_review"
    assert any("quote_not_in_source" in s for s in row["validation_errors"])


@pytest.mark.parametrize("source_text", [
    SOURCE.replace("được nghỉ", "được\nnghỉ"),
    SOURCE.replace("được nghỉ", "được\r\n\t  nghỉ"),
    SOURCE.replace("được nghỉ", "được\u00a0nghỉ"),
    unicodedata.normalize("NFD", SOURCE.replace("được nghỉ", "được\nnghỉ")),
])
def test_source_line_wrapping_does_not_invalidate_an_exact_evidence_quote(source_text):
    p = prediction()
    p.context["sources"][0]["text"] = source_text
    row = analyze_case(p, judgment(p.case_id, assessment()))
    assert row["validation_errors"] == []
    assert row["outcome"] == "answered_correctly"


@pytest.mark.parametrize("quote", [
    SOURCE.replace("Điều 5", "Điều 6"),
    SOURCE.replace("được nghỉ", "không được nghỉ"),
    SOURCE.replace("được nghỉ hằng năm", "được..."),
    SOURCE.replace("Điều 5.", "Điều 5:"),
    " \n\t ",
])
def test_whitespace_tolerance_never_accepts_changed_content_or_empty_quotes(quote):
    p = prediction()
    a = assessment()
    a.diagnostics.essential_claims[0].evidence[0].quote = quote
    row = analyze_case(p, judgment(p.case_id, a))
    assert row["outcome"] == "needs_review"
    assert "claim_0:quote_not_in_source:SOURCE_1" in row["validation_errors"]


def test_normalized_quote_must_still_belong_to_the_claimed_source():
    p = prediction()
    p.context["sources"][0]["text"] = "Một quy định khác."
    p.context["sources"].append({"source_id": "SOURCE_2", "text": SOURCE.replace("được nghỉ", "được\nnghỉ")})
    row = analyze_case(p, judgment(p.case_id, assessment()))
    assert row["outcome"] == "needs_review"
    assert "claim_0:quote_not_in_source:SOURCE_1" in row["validation_errors"]


def test_sufficient_context_requires_supported_essential_claims():
    p = prediction(abstained=True)
    a = assessment(behavior="abstain", correct="not_applicable", support="not_applicable")
    a.diagnostics.essential_claims = []
    row = analyze_case(p, judgment(p.case_id, a))
    assert row["outcome"] == "needs_review"


def test_invalid_citation_and_conflicting_low_score_cannot_be_answered_correctly():
    p = prediction()
    p.prediction.citations = [{"source_id": "SOURCE_999"}]
    row = analyze_case(p, judgment(p.case_id, assessment()))
    assert row["outcome"] == "needs_review"
    assert any("citation_ids_exist" in value for value in row["validation_errors"])
    p = prediction()
    a = assessment()
    a.citation_support = 1
    assert analyze_case(p, judgment(p.case_id, a))["outcome"] == "needs_review"


def test_primary_outcomes_sum_to_total_and_conditional_denominators_are_explicit():
    first, second, third, fourth = prediction("1", abstained=True), prediction("2"), prediction("3"), prediction("4")
    js = [judgment("1", assessment(behavior="abstain", correct="not_applicable", support="not_applicable")),
          judgment("2", assessment()),
          judgment("3", assessment(context="insufficient", support="unsupported", correct="incorrect"))]
    rows, summary = build_analysis([p.model_dump(mode="json") for p in (first, second, third, fourth)], js)
    assert sum(v["count"] for v in summary["outcomes"].values()) == 4
    rate = summary["conditional_rates"]["false_abstention_given_sufficient_qa_context"]
    assert rate == {"numerator": 1, "denominator": 2, "rate": 0.5, "case_ids": ["1"]}
    assert summary["outcomes"]["needs_review"]["count"] == 1
    assert summary["flag_counts_nonexclusive"]["incorrect_answer"] == 1
    assert summary["flag_counts_nonexclusive"]["unsupported_answer"] == 1


def test_missing_or_legacy_judge_is_review_not_an_invented_verdict(tmp_path):
    p = prediction()
    row = analyze_case(p, None)
    assert row["outcome"] == "needs_review"
    legacy = {"case_id": p.case_id, "assessment": {"overall_pass": True}}
    assert analyze_case(p, legacy)["outcome"] == "needs_review"
    path = tmp_path / "predictions.jsonl"
    append_jsonl(path, p)
    summary = write_analysis(path, None, tmp_path / "analysis")
    assert summary["classification_coverage"] == 0
    assert (tmp_path / "analysis/report.md").is_file()
    assert len((tmp_path / "analysis/cases.jsonl").read_text().splitlines()) == 1


def test_no_denominator_reports_null_not_zero():
    _, summary = build_analysis([], [])
    assert summary["conditional_rates"]["missed_clarification"]["rate"] is None


def test_reference_conflict_does_not_dilute_known_false_abstention_rate():
    first, second = prediction("1", abstained=True), prediction("2")
    _, summary = build_analysis([first.model_dump(mode="json"), second.model_dump(mode="json")], [
        judgment("1", assessment(behavior="abstain", correct="not_applicable", support="not_applicable")),
        judgment("2", assessment(conflict=True)),
    ])
    metric = summary["conditional_rates"]["false_abstention_given_sufficient_qa_context"]
    assert metric["numerator"] == metric["denominator"] == 1
    assert summary["outcomes"]["needs_review"]["count"] == 1


def test_stale_judgment_cannot_be_applied_to_changed_prediction():
    original = prediction()
    saved = judgment(original.case_id, assessment())
    saved.update(request_id=original.request_id, prediction_sha256=prediction_fingerprint(original))
    changed = original.model_copy(deep=True)
    changed.prediction.answer = "Một câu trả lời đã thay đổi."
    row = analyze_case(changed, saved)
    assert row["outcome"] == "needs_review"
    assert row["flags"] == ["judgment_prediction_mismatch"]
