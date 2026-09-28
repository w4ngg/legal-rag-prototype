from __future__ import annotations

"""Normalized failures raised by online vector-store adapters."""


class VectorStoreError(RuntimeError):
    """Base error for the online vector-store boundary."""


class QdrantUnavailableError(VectorStoreError):
    """Qdrant could not complete an online operation."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(f"Qdrant operation failed: {operation}")
