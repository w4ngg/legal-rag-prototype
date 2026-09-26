from legal_ir_prototype import (
    AppConfig,
    DenseEncoder,
    FusionStrategy,
    LegalChunk,
    QdrantVectorStore,
    Reranker,
    RetrievalService,
    SparseEncoder,
)
from legal_ir_prototype.config import AppConfig as LegacyAppConfig
from legal_ir_prototype.dense import DenseEncoder as LegacyDenseEncoder
from legal_ir_prototype.models import LegalChunk as LegacyLegalChunk
from legal_ir_prototype.service import RetrievalService as LegacyRetrievalService
from legal_ir_prototype.vector_store import (
    QdrantVectorStore as LegacyQdrantVectorStore,
)


def test_root_and_legacy_imports_reference_canonical_components() -> None:
    assert AppConfig is LegacyAppConfig
    assert DenseEncoder is LegacyDenseEncoder
    assert LegalChunk is LegacyLegalChunk
    assert RetrievalService is LegacyRetrievalService
    assert QdrantVectorStore is LegacyQdrantVectorStore
    assert FusionStrategy.__module__ == "legal_ir_prototype.fusion.weighted_rrf"
    assert Reranker.__module__ == "legal_ir_prototype.model.reranker"
    assert SparseEncoder.__module__ == "legal_ir_prototype.model.sparse"
