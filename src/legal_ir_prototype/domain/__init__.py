"""Domain entities independent of infrastructure implementations."""

from .models import LegalChunk, RetrievalHit, SparseQueryVector

__all__ = ["LegalChunk", "RetrievalHit", "SparseQueryVector"]
