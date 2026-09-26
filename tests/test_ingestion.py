from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from qdrant_client import QdrantClient

from legal_ir_prototype.core import SparseConfig
from legal_ir_prototype.ingestion.chunking import build_long_chunks
from legal_ir_prototype.ingestion.config import ChunkingConfig, IngestionConfig
from legal_ir_prototype.ingestion.qdrant import ingest_direct
from legal_ir_prototype.ingestion.schema import (
    build_payload,
    canonical_chunk_hash,
    iter_chunk_records,
)
from legal_ir_prototype.ingestion.snapshot import _copy_sparse_query_bundle
from legal_ir_prototype.model import SparseEncoder
from legal_ir_prototype.vectorstore.qdrant import QdrantVectorStore


def _write_raw(path: Path, document_id: str, passage: str) -> None:
    path.write_text(
        json.dumps(
            {
                "id": document_id,
                "name": f"Văn bản {document_id}",
                "link": f"https://example.test/{document_id}",
                "passage": passage,
                "metadata": {"legal": {"articles": ["Điều 1"]}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_ingestion_config_reuses_pinned_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    config = IngestionConfig.from_yaml(root / "configs" / "ingestion.yaml")

    assert config.runtime.dense.model_id == "AITeamVN/Vietnamese_Embedding_v2"
    assert config.runtime.dense.revision == (
        "18b44161e041bf1d3a333ab5144b5b7b93f914d2"
    )
    assert config.runtime.qdrant.dense_vector_name == "dense"
    assert config.runtime.qdrant.sparse_vector_name == "sparse"
    assert config.chunking.long_max_characters == 2000


def test_raw_chunking_is_deterministic_and_payload_compatible(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_raw(
        raw / "context_1.json",
        "1",
        "Điều 1. Phạm vi điều chỉnh\n1. Nội dung thứ nhất.\n2. Nội dung thứ hai.",
    )
    config = ChunkingConfig(
        input_pattern="context_*.json",
        short_max_characters=35,
        long_max_characters=70,
        long_overlap_short_chunks=1,
        include_document_title=True,
        title_max_characters=300,
    )
    first = build_long_chunks(raw, tmp_path / "first", config)
    second = build_long_chunks(raw, tmp_path / "second", config)

    assert canonical_chunk_hash(first) == canonical_chunk_hash(second)
    records = list(iter_chunk_records(first))
    assert records
    assert all(len(record["passage"]) <= 70 for record in records)
    payload = build_payload(records[0], vector_row=0)
    assert payload["schema_version"] == 1
    assert payload["granularity"] == "long"
    assert payload["legal"]["articles"] == ["Điều 1"]

    chunk = QdrantVectorStore._chunk_from_point(
        SimpleNamespace(id="sample", payload=payload)
    )
    assert chunk.chunk_id == records[0]["chunk_id"]
    assert chunk.metadata["legal"]["articles"] == ["Điều 1"]


def test_direct_ingestion_writes_dense_sparse_without_faiss(
    tmp_path: Path, monkeypatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = IngestionConfig.from_yaml(root / "configs" / "ingestion.yaml")
    chunks = tmp_path / "chunks.jsonl"
    records = [
        {
            "chunk_id": f"doc-{index}:long:000000",
            "document_id": f"doc-{index}",
            "passage": f"Điều {index}. Quy định pháp luật khiếu nại {index}.",
            "retrieval_text": (
                f"Tên văn bản: Luật {index}\n"
                f"Điều {index}. Quy định pháp luật khiếu nại {index}."
            ),
            "metadata": {
                "granularity": "long",
                "source_file": f"context_{index}.json",
                "chunk_index": 0,
                "chunk_count_in_document": 1,
            },
        }
        for index in range(1, 4)
    ]
    chunks.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )

    client = QdrantClient(":memory:")
    monkeypatch.setattr(
        "legal_ir_prototype.ingestion.qdrant._client", lambda runtime: client
    )

    class FakeDenseEncoder:
        def __init__(self, dense_config) -> None:
            self.dimension = dense_config.dimension

        def encode_documents(self, texts, **kwargs):
            result = np.zeros((len(texts), self.dimension), dtype=np.float32)
            for row in range(len(texts)):
                result[row, row % self.dimension] = 1.0
            return result

    monkeypatch.setattr(
        "legal_ir_prototype.ingestion.qdrant.DenseEncoder", FakeDenseEncoder
    )
    upload = replace(
        config.upload,
        batch_size=2,
        disable_indexing_during_upload=False,
        optimizer_timeout_seconds=0,
    )
    manifest_path = ingest_direct(
        chunks,
        tmp_path / "state",
        config.runtime,
        config.index,
        upload,
        collection_name="test_hybrid",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["chunk_count"] == 3
    assert client.count("test_hybrid", exact=True).count == 3
    assert not list(tmp_path.rglob("*.faiss"))
    sparse_manifest = tmp_path / "state" / "sparse_query" / "manifest.json"
    assert sparse_manifest.is_file()
    encoder = SparseEncoder(SparseConfig(sparse_manifest))
    assert encoder.encode("pháp luật khiếu nại").indices


def test_snapshot_bundle_copies_client_side_sparse_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = IngestionConfig.from_yaml(root / "configs" / "ingestion.yaml")
    source = tmp_path / "source"
    source.mkdir()
    (source / "vocab.index.json").write_text('{"luật": 0}', encoding="utf-8")
    (source / "params.index.json").write_text('{"num_docs": 1}', encoding="utf-8")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "bm25s_sparse_query_encoder",
                "collection_name": "snapshot_collection",
                "vocabulary": {"filename": "vocab.index.json"},
                "bm25": {"filename": "params.index.json"},
            }
        ),
        encoding="utf-8",
    )
    runtime = replace(
        config.runtime,
        sparse=SparseConfig(source / "manifest.json"),
    )

    result = _copy_sparse_query_bundle(
        runtime,
        tmp_path / "snapshot",
        collection_name="snapshot_collection",
    )

    assert result["manifest"] == "manifest.json"
    assert set(result["artifacts_sha256"]) == {
        "manifest.json",
        "params.index.json",
        "vocab.index.json",
    }
