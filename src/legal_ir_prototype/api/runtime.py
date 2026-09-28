from __future__ import annotations

"""Long-lived model/client ownership and dependency readiness checks."""

import inspect
import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from ..core import AppConfig
from ..domain import LLMAnswer
from ..llm import FakeLLMClient, LLMClient, OpenAICompatibleLLMClient
from ..rag import (
    ContextBuildConfig,
    ContextBuilder,
    EvidenceChecker,
    EvidencePolicy,
    RAGService,
)
from ..services import RetrievalService, build_service
from .metrics import APIMetrics
from .observability import observe_request_stage


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    embedding: bool
    reranker: bool
    qdrant: bool
    cuda: bool
    llm: bool

    @property
    def ready(self) -> bool:
        return all(
            (self.embedding, self.reranker, self.qdrant, self.cuda, self.llm)
        )

    def to_dict(self) -> dict[str, bool | str]:
        return {
            "status": "ready" if self.ready else "not_ready",
            "embedding": self.embedding,
            "reranker": self.reranker,
            "qdrant": self.qdrant,
            "cuda": self.cuda,
            "llm": self.llm,
        }


class ApplicationRuntime:
    """Own singleton inference dependencies for one API process."""

    def __init__(
        self,
        *,
        config: AppConfig,
        retrieval_service: RetrievalService,
        rag_service: RAGService,
        llm_client: LLMClient,
        require_cuda: bool = False,
    ) -> None:
        self.config = config
        self.retrieval_service = retrieval_service
        self.rag_service = rag_service
        self.llm_client = llm_client
        self.require_cuda = bool(require_cuda)

    @classmethod
    def build(
        cls,
        config: AppConfig,
        *,
        require_cuda: bool = False,
    ) -> "ApplicationRuntime":
        if config.llm is None:
            raise ValueError("API runtime requires an llm configuration section")
        retrieval_service = build_service(config)
        if config.llm.provider == "fake":
            llm_client: LLMClient = FakeLLMClient(
                LLMAnswer(
                    answer=(
                        "Đây là câu trả lời giả lập dựa trên nguồn được truy xuất."
                    ),
                    citation_ids=("SOURCE_1",),
                    model=config.llm.model,
                    confidence="high",
                )
            )
        else:
            llm_client = OpenAICompatibleLLMClient(config.llm)
        rag_config = config.rag
        evidence_config = rag_config.evidence
        return cls(
            config=config,
            retrieval_service=retrieval_service,
            rag_service=RAGService(
                retrieval_service=retrieval_service,
                llm_client=llm_client,
                context_builder=ContextBuilder(
                    ContextBuildConfig(
                        max_context_tokens=rag_config.max_context_tokens,
                        max_chunks=rag_config.max_chunks,
                        max_chunks_per_document=(
                            rag_config.max_chunks_per_document
                        ),
                        include_legal_metadata=(
                            rag_config.include_legal_metadata
                        ),
                        merge_adjacent_chunks=rag_config.merge_adjacent_chunks,
                    )
                ),
                evidence_checker=EvidenceChecker(
                    EvidencePolicy(
                        min_sources=evidence_config.min_sources,
                        min_top_reranker_score=(
                            evidence_config.min_top_reranker_score
                        ),
                        min_top_score_margin=(
                            evidence_config.min_top_score_margin
                        ),
                        abstention_answer=evidence_config.abstention_answer,
                    )
                ),
            ),
            llm_client=llm_client,
            require_cuda=require_cuda,
        )

    def startup(self) -> None:
        self.retrieval_service.dense_encoder.load()
        self.retrieval_service.reranker.load()
        self.retrieval_service.vector_store.validate_schema(
            expected_dense_dimension=self.config.dense.dimension
        )

    def bind_metrics(self, metrics: APIMetrics) -> None:
        def observe_retrieval_component(stage: str, duration: float) -> None:
            metrics.observe_retrieval_component(stage, duration)
            observe_request_stage(stage, duration)

        self.retrieval_service.set_timing_observer(observe_retrieval_component)
        self.rag_service.set_timing_observer(observe_request_stage)

    def readiness(self) -> ReadinessReport:
        qdrant_ready = False
        try:
            self.retrieval_service.vector_store.check_connection()
            qdrant_ready = True
        except Exception:
            qdrant_ready = False

        cuda_ready = True
        if self.require_cuda:
            try:
                import torch

                cuda_ready = bool(torch.cuda.is_available())
            except Exception:
                cuda_ready = False
        return ReadinessReport(
            embedding=self.retrieval_service.dense_encoder.is_loaded,
            reranker=self.retrieval_service.reranker.is_loaded,
            qdrant=qdrant_ready,
            cuda=cuda_ready,
            llm=self.config.llm is not None,
        )

    async def aclose(self) -> None:
        close = getattr(self.llm_client, "aclose", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

    def version_info(self) -> dict[str, Any]:
        try:
            service_version = version("legal-ir-prototype")
        except PackageNotFoundError:
            service_version = "development"
        assert self.config.llm is not None
        return {
            "service_version": service_version,
            "release_id": os.environ.get("LEGAL_RAG_RELEASE_ID", "development"),
            "git_sha": os.environ.get("LEGAL_RAG_GIT_SHA"),
            "dense_model": self.config.dense.model_id,
            "dense_revision": self.config.dense.revision,
            "reranker_model": self.config.reranker.model_id,
            "reranker_revision": self.config.reranker.revision,
            "qdrant_collection": self.config.qdrant.collection,
            "llm_model": self.config.llm.model,
            "prompt_version": "legal_qa_v2",
            "output_schema_version": 1,
        }
