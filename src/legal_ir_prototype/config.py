"""Backward-compatible imports for configuration types."""

from .core.config import (
    AppConfig,
    DenseConfig,
    FusionConfig,
    QdrantConfig,
    RerankerConfig,
    RetrievalConfig,
    SparseConfig,
)

__all__ = [
    "AppConfig",
    "DenseConfig",
    "FusionConfig",
    "QdrantConfig",
    "RerankerConfig",
    "RetrievalConfig",
    "SparseConfig",
]
