"""Grounded context construction for legal question answering."""

from .abstention import EvidenceChecker, EvidenceDecision, EvidencePolicy
from .citation import (
    CitationValidationError,
    CitationValidationResult,
    CitationValidator,
)
from .context_builder import (
    ApproximateTokenCounter,
    ContextBuildConfig,
    ContextBuilder,
    TokenCounter,
)
from .prompt_builder import LEGAL_QA_SYSTEM_PROMPT, LLMPrompt, PromptBuilder
from .service import RAGExecution, RAGService, Retriever

__all__ = [
    "ApproximateTokenCounter",
    "ContextBuildConfig",
    "ContextBuilder",
    "CitationValidationError",
    "CitationValidationResult",
    "CitationValidator",
    "EvidenceChecker",
    "EvidenceDecision",
    "EvidencePolicy",
    "LEGAL_QA_SYSTEM_PROMPT",
    "LLMPrompt",
    "PromptBuilder",
    "RAGExecution",
    "RAGService",
    "Retriever",
    "TokenCounter",
]
