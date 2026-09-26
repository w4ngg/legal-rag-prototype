from __future__ import annotations

from pathlib import Path

from legal_ir_prototype.core import (
    AppConfig,
    DenseConfig,
    FusionConfig,
    QdrantConfig,
    RerankerConfig,
    RetrievalConfig,
    SparseConfig,
)
from legal_ir_prototype.domain import LegalChunk, RetrievalHit, SparseQueryVector
from legal_ir_prototype.fusion import FusionStrategy
from legal_ir_prototype.services import RetrievalService


def _config() -> AppConfig:
    return AppConfig(
        dense=DenseConfig(
            "embedding", "e" * 40, 2048, 3, 2, True, "cpu", "float32", False
        ),
        sparse=SparseConfig(Path("manifest.json")),
        qdrant=QdrantConfig(
            "http://localhost:6333",
            None,
            "collection",
            "dense",
            "sparse",
            False,
            6334,
            30,
            True,
            128,
        ),
        retrieval=RetrievalConfig(2, 2),
        fusion=FusionConfig("weighted_rrf", 60, 1.0, 1.0, 120, 120),
        reranker=RerankerConfig(
            "reranker", "f" * 40, 2304, 4, 2, "cpu", "float32", False
        ),
        source_path=Path("runtime.yaml"),
    )


def _chunk(chunk_id: str) -> LegalChunk:
    return LegalChunk(chunk_id, chunk_id[0], f"p-{chunk_id}", f"t-{chunk_id}")


class FakeDense:
    def encode(self, query: str) -> list[float]:
        assert query == "query"
        return [1.0, 2.0, 3.0]


class FakeSparse:
    vector_name = "sparse"

    def encode(self, query: str) -> SparseQueryVector:
        assert query == "query"
        return SparseQueryVector((2,), (1.0,))


class FakeStore:
    def validate_schema(self, *, expected_dense_dimension: int) -> None:
        assert expected_dense_dimension == 3

    def search_dense(self, vector: object, top_k: int) -> list[RetrievalHit]:
        assert list(vector) == [1.0, 2.0, 3.0]
        assert top_k == 2
        return [
            RetrievalHit(_chunk("a:0"), "dense", 0.9),
            RetrievalHit(_chunk("b:0"), "dense", 0.8),
        ]

    def search_sparse(self, vector: object, top_k: int) -> list[RetrievalHit]:
        assert top_k == 2
        return [
            RetrievalHit(_chunk("c:0"), "sparse", 7.0),
            RetrievalHit(_chunk("a:0"), "sparse", 6.0),
        ]


class FakeReranker:
    def score(self, query: str, chunks: list[LegalChunk]) -> list[float]:
        assert query == "query"
        assert [chunk.chunk_id for chunk in chunks] == ["a:0", "c:0", "b:0"]
        return [0.0, 3.0, 2.0]


def test_service_ranks_only_by_reranker_after_full_union() -> None:
    config = _config()
    service = RetrievalService(
        config=config,
        dense_encoder=FakeDense(),  # type: ignore[arg-type]
        sparse_encoder=FakeSparse(),  # type: ignore[arg-type]
        vector_store=FakeStore(),  # type: ignore[arg-type]
        fusion=FusionStrategy(config.fusion),
        reranker=FakeReranker(),  # type: ignore[arg-type]
    )

    results = service.retrieve(" query ")

    assert [result.chunk_id for result in results] == ["c:0", "b:0"]
    assert [result.reranker_score for result in results] == [3.0, 2.0]
    assert results[0].fusion_rank == 2
    assert results[0].reranker_rank == 1
