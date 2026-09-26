from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from legal_ir_prototype.core import AppConfig
from legal_ir_prototype.model import DenseEncoder, SparseEncoder
from legal_ir_prototype.vectorstore import QdrantVectorStore


@pytest.mark.integration
def test_live_representative_queries_against_research_retrieval_output() -> None:
    """Opt-in exact lane parity test against preserved research diagnostics.

    Set the two environment variables after starting Qdrant. The fixture is not
    bundled because the prototype intentionally excludes experiment artifacts.
    """

    config_value = os.environ.get("LEGAL_IR_PROTOTYPE_CONFIG")
    diagnostics_value = os.environ.get("LEGAL_IR_RESEARCH_DEEP_DIAGNOSTICS")
    if not config_value or not diagnostics_value:
        pytest.skip(
            "set LEGAL_IR_PROTOTYPE_CONFIG and "
            "LEGAL_IR_RESEARCH_DEEP_DIAGNOSTICS"
        )
    config = AppConfig.from_yaml(config_value)
    diagnostics_path = Path(diagnostics_value)
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    queries = diagnostics.get("queries") or {}
    representative_ids = ("83864", "151910", "154226")
    dense_encoder = DenseEncoder(config.dense)
    sparse_encoder = SparseEncoder(config.sparse)
    vector_store = QdrantVectorStore(config.qdrant)
    vector_store.validate_schema(expected_dense_dimension=config.dense.dimension)

    for query_id in representative_ids:
        record = queries.get(query_id)
        if not isinstance(record, dict):
            pytest.fail(f"research diagnostics is missing query {query_id}")
        query = str(record["query"])
        expected_channels = record.get("channels") or {}
        dense_hits = vector_store.search_dense(
            dense_encoder.encode(query), config.retrieval.dense_top_k
        )
        sparse_hits = vector_store.search_sparse(
            sparse_encoder.encode(query), config.retrieval.sparse_top_k
        )
        expected_dense = expected_channels["dense"]["chunk_hits"]
        expected_sparse = expected_channels["bm25"]["chunk_hits"]
        assert [hit.chunk.chunk_id for hit in dense_hits] == [
            str(hit["chunk_id"])
            for hit in expected_dense[: config.retrieval.dense_top_k]
        ]
        assert [hit.chunk.chunk_id for hit in sparse_hits] == [
            str(hit["chunk_id"])
            for hit in expected_sparse[: config.retrieval.sparse_top_k]
        ]
