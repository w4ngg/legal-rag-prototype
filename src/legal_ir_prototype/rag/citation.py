from __future__ import annotations

"""Resolve untrusted LLM citation IDs against server-owned context sources."""

from dataclasses import dataclass

from ..domain import BuiltContext, Citation, LLMAnswer


@dataclass(frozen=True, slots=True)
class CitationValidationResult:
    """Result of resolving one LLM answer against one immutable context."""

    citations: tuple[Citation, ...]
    invalid_source_ids: tuple[str, ...] = ()
    missing_required_citations: bool = False

    @property
    def is_valid(self) -> bool:
        return not self.invalid_source_ids and not self.missing_required_citations

    @property
    def errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.invalid_source_ids:
            errors.append(
                "unknown_citation_ids=" + ",".join(self.invalid_source_ids)
            )
        if self.missing_required_citations:
            errors.append("answer_requires_at_least_one_valid_citation")
        return tuple(errors)


class CitationValidationError(ValueError):
    """Raised when strict citation validation is requested and fails."""

    def __init__(self, result: CitationValidationResult) -> None:
        self.result = result
        super().__init__("; ".join(result.errors) or "citation validation failed")


class CitationValidator:
    """Validate IDs and hydrate citation metadata from ``BuiltContext`` only."""

    def __init__(self, *, require_citations_for_answer: bool = True) -> None:
        self.require_citations_for_answer = bool(require_citations_for_answer)

    def validate(
        self,
        *,
        answer: LLMAnswer,
        context: BuiltContext,
    ) -> CitationValidationResult:
        sources = {source.source_id: source for source in context.sources}
        requested_ids = tuple(dict.fromkeys(answer.citation_ids))
        invalid_ids = tuple(
            source_id for source_id in requested_ids if source_id not in sources
        )
        citations = tuple(
            Citation(
                source_id=source.source_id,
                document_id=source.document_id,
                chunk_ids=source.chunk_ids,
                metadata=source.metadata,
            )
            for source_id in requested_ids
            if (source := sources.get(source_id)) is not None
        )
        missing_required = (
            self.require_citations_for_answer
            and not answer.insufficient_evidence
            and not citations
        )
        return CitationValidationResult(
            citations=citations,
            invalid_source_ids=invalid_ids,
            missing_required_citations=missing_required,
        )

    def validate_or_raise(
        self,
        *,
        answer: LLMAnswer,
        context: BuiltContext,
    ) -> tuple[Citation, ...]:
        result = self.validate(answer=answer, context=context)
        if not result.is_valid:
            raise CitationValidationError(result)
        return result.citations
