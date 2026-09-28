from __future__ import annotations

"""Per-application Prometheus registry and bounded-cardinality metrics."""

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from ..rag import RAGExecution


class APIMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.rag_requests = Counter(
            "rag_requests",
            "Completed RAG requests by outcome.",
            ("outcome",),
            registry=self.registry,
        )
        self.rag_request_errors = Counter(
            "rag_request_errors",
            "RAG/API errors by stable error code.",
            ("code",),
            registry=self.registry,
        )
        self.rag_request_duration = Histogram(
            "rag_request_duration_seconds",
            "End-to-end RAG request duration.",
            registry=self.registry,
        )
        self.rag_stage_duration = Histogram(
            "rag_stage_duration_seconds",
            "RAG stage duration.",
            ("stage",),
            registry=self.registry,
        )
        self.retrieval_requests = Counter(
            "retrieval_requests",
            "Internal retrieval requests by outcome.",
            ("outcome",),
            registry=self.registry,
        )
        self.retrieval_duration = Histogram(
            "retrieval_duration_seconds",
            "Internal retrieval endpoint duration.",
            registry=self.registry,
        )
        self.reranker_duration = Histogram(
            "reranker_duration_seconds",
            "Cross-encoder reranker duration.",
            registry=self.registry,
        )
        self.llm_duration = Histogram(
            "llm_duration_seconds",
            "LLM provider request duration.",
            registry=self.registry,
        )
        self.qdrant_request_duration = Histogram(
            "qdrant_request_duration_seconds",
            "Qdrant vector search duration by channel.",
            ("channel",),
            registry=self.registry,
        )
        self.llm_requests = Counter(
            "llm_requests",
            "LLM calls initiated by the RAG service.",
            registry=self.registry,
        )
        self.rag_abstentions = Counter(
            "rag_abstentions",
            "RAG abstentions by stable reason.",
            ("reason",),
            registry=self.registry,
        )

    def observe_rag(self, execution: RAGExecution) -> None:
        response = execution.response
        outcome = "abstained" if response.abstained else "answered"
        self.rag_requests.labels(outcome=outcome).inc()
        total = execution.timings_seconds.get("total")
        if total is not None:
            self.rag_request_duration.observe(total)
        for stage, duration in execution.timings_seconds.items():
            if stage != "total":
                self.rag_stage_duration.labels(stage=stage).observe(duration)
        retrieval = execution.timings_seconds.get("retrieval")
        if retrieval is not None:
            self.retrieval_duration.observe(retrieval)
        llm = execution.timings_seconds.get("llm")
        if llm is not None:
            self.llm_duration.observe(llm)
        if execution.prompt is not None:
            self.llm_requests.inc()
        if response.abstained:
            self.rag_abstentions.labels(
                reason=response.abstention_reason or "unspecified"
            ).inc()

    def observe_retrieval_component(self, stage: str, duration: float) -> None:
        if stage == "reranker":
            self.reranker_duration.observe(duration)
            return
        if stage.startswith("qdrant_"):
            self.qdrant_request_duration.labels(
                channel=stage.removeprefix("qdrant_")
            ).observe(duration)

    def render(self) -> bytes:
        return generate_latest(self.registry)
