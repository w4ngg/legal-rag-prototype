from __future__ import annotations

"""End-to-end grounded RAG orchestration independent of API transport."""

import asyncio
import math
import time
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Callable
from typing import Mapping, Protocol, Sequence

from ..domain import BuiltContext, LegalChunk, RAGResponse
from ..llm import LLMClient
from .abstention import EvidenceChecker, EvidenceDecision
from .citation import CitationValidator
from .context_builder import ContextBuilder
from .prompt_builder import LLMPrompt, PromptBuilder


class Retriever(Protocol):
    def retrieve(self, query: str) -> Sequence[LegalChunk]:
        """Return ranked chunks for one normalized query."""
        ...


@dataclass(frozen=True, slots=True)
class RAGExecution:
    """One RAG result plus opt-in debug values produced by the same run."""

    response: RAGResponse
    context: BuiltContext
    prompt: LLMPrompt | None
    timings_seconds: Mapping[str, float]

    def __post_init__(self) -> None:
        timings = {
            str(stage): float(duration)
            for stage, duration in self.timings_seconds.items()
        }
        if any(
            not stage or not math.isfinite(duration) or duration < 0
            for stage, duration in timings.items()
        ):
            raise ValueError("RAG timings must be finite non-negative values")
        object.__setattr__(self, "timings_seconds", MappingProxyType(timings))


class RAGService:
    """Retrieve evidence, generate an answer, and enforce grounding guards."""

    def __init__(
        self,
        *,
        retrieval_service: Retriever,
        llm_client: LLMClient,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
        citation_validator: CitationValidator | None = None,
        evidence_checker: EvidenceChecker | None = None,
        timing_observer: Callable[[str, float], None] | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm_client = llm_client
        self.citation_validator = citation_validator or CitationValidator()
        self.evidence_checker = evidence_checker or EvidenceChecker()
        self._timing_observer = timing_observer

    def set_timing_observer(
        self,
        observer: Callable[[str, float], None] | None,
    ) -> None:
        self._timing_observer = observer

    def _observe(self, stage: str, duration: float) -> None:
        if self._timing_observer is not None:
            self._timing_observer(stage, duration)

    def _abstain(
        self,
        *,
        query: str,
        model: str,
        decision: EvidenceDecision,
    ) -> RAGResponse:
        if decision.sufficient or decision.reason is None:
            raise ValueError("abstention requires an insufficient evidence decision")
        return RAGResponse(
            query=query,
            answer=self.evidence_checker.policy.abstention_answer,
            citations=(),
            model=model,
            abstained=True,
            abstention_reason=decision.reason,
            warnings=decision.warnings,
        )

    async def execute(self, query: str) -> RAGExecution:
        """Run one query and retain the exact context/prompt for smoke tracing."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        total_started = time.perf_counter()
        timings: dict[str, float] = {}
        retrieval_started = time.perf_counter()
        try:
            chunks = tuple(
                await asyncio.to_thread(
                    self.retrieval_service.retrieve,
                    normalized_query,
                )
            )
        finally:
            retrieval_duration = time.perf_counter() - retrieval_started
            timings["retrieval"] = retrieval_duration
            self._observe("retrieval", retrieval_duration)
        context_started = time.perf_counter()
        try:
            context = self.context_builder.build(
                query=normalized_query,
                chunks=chunks,
            )
        finally:
            context_duration = time.perf_counter() - context_started
            timings["context_builder"] = context_duration
            self._observe("context_builder", context_duration)
        context_decision = self.evidence_checker.check_context(context)
        if not context_decision.sufficient:
            timings["total"] = time.perf_counter() - total_started
            return RAGExecution(
                response=self._abstain(
                    query=normalized_query,
                    model=self.llm_client.model,
                    decision=context_decision,
                ),
                context=context,
                prompt=None,
                timings_seconds=timings,
            )

        prompt = self.prompt_builder.build(context)
        llm_started = time.perf_counter()
        try:
            llm_answer = await self.llm_client.generate(
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.user_prompt,
            )
        finally:
            llm_duration = time.perf_counter() - llm_started
            timings["llm"] = llm_duration
            self._observe("llm", llm_duration)
        validation_started = time.perf_counter()
        try:
            citation_result = self.citation_validator.validate(
                answer=llm_answer,
                context=context,
            )
            answer_decision = self.evidence_checker.check_answer(
                answer=llm_answer,
                citations=citation_result,
            )
        finally:
            validation_duration = time.perf_counter() - validation_started
            timings["validation"] = validation_duration
            self._observe("validation", validation_duration)
        timings["total"] = time.perf_counter() - total_started
        if not answer_decision.sufficient:
            return RAGExecution(
                response=self._abstain(
                    query=normalized_query,
                    model=llm_answer.model,
                    decision=answer_decision,
                ),
                context=context,
                prompt=prompt,
                timings_seconds=timings,
            )

        return RAGExecution(
            response=RAGResponse(
                query=normalized_query,
                answer=llm_answer.answer,
                citations=citation_result.citations,
                model=llm_answer.model,
            ),
            context=context,
            prompt=prompt,
            timings_seconds=timings,
        )

    async def answer(self, query: str) -> RAGResponse:
        return (await self.execute(query)).response

    async def generate(self, query: str) -> RAGResponse:
        """Alias for callers that use generation-oriented naming."""

        return await self.answer(query)
