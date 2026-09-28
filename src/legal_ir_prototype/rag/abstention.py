from __future__ import annotations

"""Evidence sufficiency policy for grounded answer generation."""

import math
from dataclasses import dataclass

from ..domain import BuiltContext, LLMAnswer
from .citation import CitationValidationResult


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    """Calibratable evidence gates; score gates are disabled by default."""

    min_sources: int = 1
    min_top_reranker_score: float | None = None
    min_top_score_margin: float | None = None
    abstention_answer: str = (
        "Không đủ bằng chứng trong các nguồn pháp luật được truy xuất để trả lời."
    )

    def __post_init__(self) -> None:
        if isinstance(self.min_sources, bool) or int(self.min_sources) <= 0:
            raise ValueError("min_sources must be a positive integer")
        object.__setattr__(self, "min_sources", int(self.min_sources))
        for field_name in (
            "min_top_reranker_score",
            "min_top_score_margin",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"{field_name} must be finite")
            if field_name == "min_top_score_margin" and normalized < 0:
                raise ValueError("min_top_score_margin must be non-negative")
            object.__setattr__(self, field_name, normalized)
        normalized_answer = self.abstention_answer.strip()
        if not normalized_answer:
            raise ValueError("abstention_answer must not be empty")
        object.__setattr__(self, "abstention_answer", normalized_answer)


@dataclass(frozen=True, slots=True)
class EvidenceDecision:
    sufficient: bool
    reason: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        reason = None if self.reason is None else self.reason.strip()
        if self.sufficient and reason is not None:
            raise ValueError("a sufficient decision cannot have a reason")
        if not self.sufficient and not reason:
            raise ValueError("an insufficient decision requires a reason")
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "warnings",
            tuple(str(value).strip() for value in self.warnings if str(value).strip()),
        )


class EvidenceChecker:
    """Apply deterministic pre-generation and post-generation evidence gates."""

    def __init__(self, policy: EvidencePolicy | None = None) -> None:
        self.policy = policy or EvidencePolicy()

    def check_context(self, context: BuiltContext) -> EvidenceDecision:
        if len(context.sources) < self.policy.min_sources:
            return EvidenceDecision(False, "insufficient_context_sources")

        scores = sorted(
            (
                source.reranker_score
                for source in context.sources
                if source.reranker_score is not None
            ),
            reverse=True,
        )
        threshold = self.policy.min_top_reranker_score
        if threshold is not None:
            if not scores:
                return EvidenceDecision(False, "missing_reranker_scores")
            if scores[0] < threshold:
                return EvidenceDecision(False, "top_reranker_score_below_threshold")

        margin = self.policy.min_top_score_margin
        if margin is not None and len(scores) >= 2 and scores[0] - scores[1] < margin:
            return EvidenceDecision(False, "top_reranker_margin_below_threshold")
        return EvidenceDecision(True)

    def check_answer(
        self,
        *,
        answer: LLMAnswer,
        citations: CitationValidationResult,
    ) -> EvidenceDecision:
        if not citations.is_valid:
            return EvidenceDecision(
                False,
                "citation_validation_failed",
                citations.errors,
            )
        if answer.insufficient_evidence:
            return EvidenceDecision(False, "llm_reported_insufficient_evidence")
        return EvidenceDecision(True)
