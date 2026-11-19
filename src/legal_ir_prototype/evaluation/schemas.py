from __future__ import annotations

"""Versioned schemas written by the offline RAG evaluation commands."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _required_text(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must not be empty")
    return normalized


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    case_id: str
    source_query_id: str
    question: str
    reference_answer: str
    reference_quality: Literal["unreviewed", "accepted"] = "unreviewed"
    tags: tuple[str, ...] = ()
    source_file: str | None = None
    expected_behavior: Literal["answer", "abstain", "clarify", "safe_response"] = "answer"

    _case_id = field_validator("case_id")(_required_text)
    _source_query_id = field_validator("source_query_id")(_required_text)
    _question = field_validator("question")(_required_text)
    _reference_answer = field_validator("reference_answer")(_required_text)

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(value.strip() for value in values if value.strip()))
        return normalized


class ExcludedSourceCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_query_id: str
    reason: str

    _source_query_id = field_validator("source_query_id")(_required_text)
    _reason = field_validator("reason")(_required_text)


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    dataset_id: str
    created_at: str
    source_path: str
    source_sha256: str
    selection_method: Literal["seeded_random_without_replacement", "stratified_unique_questions", "curated_guardrails", "combined_suite"]
    seed: int
    requested_sample_size: int = Field(gt=0)
    selected_sample_size: int = Field(gt=0)
    selected_source_query_ids: tuple[str, ...]
    excluded_cases: tuple[ExcludedSourceCase, ...] = ()
    sources: tuple[dict[str, Any], ...] = ()
    slice_counts: dict[str, int] = Field(default_factory=dict)
    cases_sha256: str | None = None


class PredictionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[dict[str, Any]]
    abstained: bool
    abstention_reason: str | None = None
    warnings: list[str]
    model: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    finish_reason: str | None = None

    _answer = field_validator("answer")(_required_text)
    _model = field_validator("model")(_required_text)


class EvaluationPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    request_id: str
    case_id: str
    source_query_id: str
    question: str
    reference_answer: str
    prediction: PredictionPayload
    retrieved_chunks: list[dict[str, Any]]
    context: dict[str, Any]
    timings_ms: dict[str, float]
    prompt: dict[str, str] | None = None
    raw_llm_response: str | None = None
    error: None = None
    tags: tuple[str, ...] = ()
    source_file: str | None = None
    expected_behavior: Literal["answer", "abstain", "clarify", "safe_response"] = "answer"

    _run_id = field_validator("run_id")(_required_text)
    _request_id = field_validator("request_id")(_required_text)
    _case_id = field_validator("case_id")(_required_text)


class EvaluationError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    request_id: str
    case_id: str
    source_query_id: str
    question: str
    reference_answer: str
    error_type: str
    error_message: str
    timings_ms: dict[str, float]


class DeterministicEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    checks: dict[str, bool]
    errors: tuple[str, ...] = ()


class JudgeAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correctness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    citation_support: int = Field(ge=1, le=5)
    legal_usefulness: int = Field(ge=1, le=5)
    overall_pass: bool
    rationale: str

    _rationale = field_validator("rationale")(_required_text)


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    case_id: str
    request_id: str
    deterministic: DeterministicEvaluation
    judge_model: str | None = None
    judge_prompt_version: str
    assessment: JudgeAssessment | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    error: str | None = None
    latency_ms: float | None = None
