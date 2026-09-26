"""Inference-only Vietnamese legal hybrid retrieval components."""

from .core import AppConfig
from .domain import LegalChunk, RetrievalHit, SparseQueryVector
from .fusion import FusionStrategy
from .model import DenseEncoder, Reranker, SparseEncoder
from .services import RetrievalService, build_service
from .vectorstore import QdrantVectorStore

__all__ = [
    "AppConfig",
    "DenseEncoder",
    "FusionStrategy",
    "LegalChunk",
    "QdrantVectorStore",
    "Reranker",
    "RetrievalHit",
    "RetrievalService",
    "SparseEncoder",
    "SparseQueryVector",
    "build_service",
]
