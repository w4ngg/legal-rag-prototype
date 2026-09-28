"""FastAPI serving layer for the online legal RAG pipeline."""

from .app import create_app
from .runtime import ApplicationRuntime, ReadinessReport

__all__ = ["ApplicationRuntime", "ReadinessReport", "create_app"]
