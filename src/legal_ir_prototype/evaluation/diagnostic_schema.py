from __future__ import annotations

"""Evidence and behavior judgments used by the diagnostic evaluator (v3)."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .schemas import EvaluationPrediction, JudgeAssessment, JudgeResult


class EvidenceQuote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class EssentialClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(min_length=1)
    support: Literal["supported", "unsupported", "unknown"]
    evidence: list[EvidenceQuote]


class EvidenceDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context_sufficiency: Literal["sufficient", "partial", "insufficient", "conflicting", "unknown"]
    actual_behavior: Literal["answer", "abstain", "clarify", "refuse", "mixed", "unknown"]
    answer_correctness: Literal["correct", "partially_correct", "incorrect", "not_applicable", "unknown"]
    answer_support: Literal["supported", "partially_supported", "unsupported", "not_applicable", "unknown"]
    requires_clarification: bool | None
    behavior_pass: bool | None
    guardrail_violation: bool | None
    reference_conflict: bool
    legal_basis_relation: Literal["match", "equivalent", "partial", "mismatch", "not_applicable", "unknown"]
    essential_claims: list[EssentialClaim]
    missing_information: list[str]


class DiagnosticAssessment(JudgeAssessment):
    diagnostics: EvidenceDiagnostics


class DiagnosticItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    assessment: DiagnosticAssessment


class DiagnosticBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[DiagnosticItem]


class DiagnosticJudgeResult(JudgeResult):
    assessment: DiagnosticAssessment | None = None
    judge_request_id: str
    judge_request_case_count: int = Field(gt=0)
    request_input_tokens: int | None = Field(default=None, ge=0)
    request_output_tokens: int | None = Field(default=None, ge=0)
    diagnostic_validation_errors: tuple[str, ...] = ()
    prediction_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def prediction_fingerprint(prediction: EvaluationPrediction) -> str:
    payload = json.dumps(prediction.model_dump(mode="json"), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
