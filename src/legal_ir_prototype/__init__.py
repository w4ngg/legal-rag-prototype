"""Inference-only Vietnamese legal hybrid retrieval components."""

from .core import AppConfig, EvidenceConfig, LLMConfig, RAGConfig
from .domain import (
    BuiltContext,
    Citation,
    ContextExclusionReason,
    ContextSource,
    ExcludedChunk,
    LegalChunk,
    LLMAnswer,
    RAGResponse,
    RetrievalHit,
    SparseQueryVector,
)
from .fusion import FusionStrategy
from .llm import FakeLLMClient, LLMClient, OpenAICompatibleLLMClient
from .model import DenseEncoder, Reranker, SparseEncoder
from .rag import (
    ApproximateTokenCounter,
    CitationValidationError,
    CitationValidationResult,
    CitationValidator,
    ContextBuildConfig,
    ContextBuilder,
    EvidenceChecker,
    EvidenceDecision,
    EvidencePolicy,
    LLMPrompt,
    PromptBuilder,
    RAGExecution,
    RAGService,
    Retriever,
)
from .services import RetrievalService, build_service
from .vectorstore import QdrantUnavailableError, QdrantVectorStore, VectorStoreError

__all__ = [
    "AppConfig",
    "ApproximateTokenCounter",
    "BuiltContext",
    "Citation",
    "CitationValidationError",
    "CitationValidationResult",
    "CitationValidator",
    "ContextBuildConfig",
    "ContextBuilder",
    "ContextExclusionReason",
    "ContextSource",
    "DenseEncoder",
    "ExcludedChunk",
    "EvidenceChecker",
    "EvidenceConfig",
    "EvidenceDecision",
    "EvidencePolicy",
    "FusionStrategy",
    "FakeLLMClient",
    "LegalChunk",
    "LLMAnswer",
    "LLMClient",
    "LLMConfig",
    "LLMPrompt",
    "OpenAICompatibleLLMClient",
    "PromptBuilder",
    "QdrantUnavailableError",
    "QdrantVectorStore",
    "RAGExecution",
    "RAGConfig",
    "RAGService",
    "Reranker",
    "RAGResponse",
    "RetrievalHit",
    "RetrievalService",
    "Retriever",
    "SparseEncoder",
    "SparseQueryVector",
    "VectorStoreError",
    "build_service",
]
