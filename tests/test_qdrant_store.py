from __future__ import annotations

import pytest
from qdrant_client import QdrantClient, models

from legal_ir_prototype.core import QdrantConfig
from legal_ir_prototype.domain import SparseQueryVector
from legal_ir_prototype.vectorstore import QdrantVectorStore


@pytest.mark.filterwarnings("ignore:Local mode performs exact")
def test_named_dense_and_sparse_vectors_return_the_same_chunk_payload() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        "legal",
        vectors_config={
            "dense_vector": models.VectorParams(
                size=3,
                distance=models.Distance.DOT,
            )
        },
        sparse_vectors_config={
            "bm25_vector": models.SparseVectorParams()
        },
    )
    client.upsert(
        "legal",
        [
            models.PointStruct(
                id=1,
                vector={
                    "dense_vector": [1.0, 0.0, 0.0],
                    "bm25_vector": models.SparseVector(
                        indices=[7], values=[2.0]
                    ),
                },
                payload={
                    "schema_version": 1,
                    "chunk_id": "doc:long:0",
                    "document_id": "doc",
                    "vector_row": 0,
                    "granularity": "long",
                    "passage": "Nội dung điều luật",
                    "retrieval_text": "Tên văn bản\nNội dung điều luật",
                    "passage_sha256": "p",
                    "embedding_text_sha256": "e",
                    "legal": {"articles": ["Điều 1"]},
                },
            )
        ],
    )
    config = QdrantConfig(
        url="http://unused",
        api_key_env=None,
        collection="legal",
        dense_vector_name="dense_vector",
        sparse_vector_name="bm25_vector",
        prefer_grpc=False,
        grpc_port=6334,
        timeout_seconds=30,
        exact=True,
        hnsw_ef=128,
    )
    store = QdrantVectorStore(config, client=client)

    store.validate_schema(expected_dense_dimension=3)
    dense = store.search_dense([1.0, 0.0, 0.0], 1)
    sparse = store.search_sparse(SparseQueryVector((7,), (1.0,)), 1)

    assert dense[0].chunk.chunk_id == sparse[0].chunk.chunk_id == "doc:long:0"
    assert dense[0].score == 1.0
    assert sparse[0].score == 2.0
    assert dense[0].chunk.metadata["legal"] == {"articles": ["Điều 1"]}
