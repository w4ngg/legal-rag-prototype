"""Domain entities independent of infrastructure implementations."""

from .context import (
    BuiltContext,
    ContextExclusionReason,
    ContextSource,
    ExcludedChunk,
)
from .models import LegalChunk, RetrievalHit, SparseQueryVector
from .rag import Citation, LLMAnswer, RAGResponse

__all__ = [
    "BuiltContext",
    "Citation",
    "ContextExclusionReason",
    "ContextSource",
    "ExcludedChunk",
    "LegalChunk",
    "LLMAnswer",
    "RAGResponse",
    "RetrievalHit",
    "SparseQueryVector",
]
