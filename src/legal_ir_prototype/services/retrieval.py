from __future__ import annotations

from ..core.config import AppConfig
from ..domain.models import LegalChunk
from ..fusion import FusionStrategy
from ..model import DenseEncoder, Reranker, SparseEncoder
from ..vectorstore import QdrantVectorStore


class RetrievalService:
    """Online long-chunk retrieval: BM25+dense union, RRF, cross-encoder."""

    def __init__(
        self,
        *,
        config: AppConfig,
        dense_encoder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        vector_store: QdrantVectorStore,
        fusion: FusionStrategy,
        reranker: Reranker,
    ) -> None:
        if sparse_encoder.vector_name != config.qdrant.sparse_vector_name:
            raise ValueError(
                "sparse query bundle vector name does not match Qdrant config"
            )
        self.config = config
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.vector_store = vector_store
        self.fusion = fusion
        self.reranker = reranker

    def retrieve(self, query: str) -> list[LegalChunk]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        self.vector_store.validate_schema(
            expected_dense_dimension=self.config.dense.dimension
        )
        dense_vector = self.dense_encoder.encode(normalized_query)
        sparse_vector = self.sparse_encoder.encode(normalized_query)
        dense_hits = self.vector_store.search_dense(
            dense_vector,
            self.config.retrieval.dense_top_k,
        )
        sparse_hits = self.vector_store.search_sparse(
            sparse_vector,
            self.config.retrieval.sparse_top_k,
        )
        candidates = self.fusion.fuse(
            {"dense": dense_hits, "sparse": sparse_hits},
        )
        reranker_scores = self.reranker.score(normalized_query, candidates)
        ranked = sorted(
            zip(candidates, reranker_scores, strict=True),
            key=lambda item: (-item[1], item[0].fusion_rank, item[0].chunk_id),
        )
        # Deliberately no MaxP document aggregation and no linear/rank blend of
        # retrieval and reranker scores. Fusion rank is only a deterministic
        # tie-break after the raw cross-encoder logit.
        return [
            chunk.with_scores(reranker_score=score, reranker_rank=rank)
            for rank, (chunk, score) in enumerate(ranked, start=1)
        ][: self.config.reranker.output_top_k]


def build_service(config: AppConfig) -> RetrievalService:
    dense_encoder = DenseEncoder(config.dense)
    sparse_encoder = SparseEncoder(config.sparse)
    vector_store = QdrantVectorStore(config.qdrant)
    return RetrievalService(
        config=config,
        dense_encoder=dense_encoder,
        sparse_encoder=sparse_encoder,
        vector_store=vector_store,
        fusion=FusionStrategy(config.fusion),
        reranker=Reranker(config.reranker),
    )
