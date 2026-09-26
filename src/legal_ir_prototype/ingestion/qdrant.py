from __future__ import annotations

import itertools
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..core import AppConfig
from ..model import DenseEncoder
from .config import IndexConfig, UploadConfig
from .schema import (
    PAYLOAD_SCHEMA,
    PAYLOAD_SCHEMA_VERSION,
    build_payload,
    canonical_chunk_hash,
    iter_chunk_records,
    normalize_chunk_record,
    point_id_for_chunk,
    sha256_file,
)
from .sparse_index import export_query_bundle, prepare_sparse_corpus


FORMAT_VERSION = 1


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        result = value.model_dump(mode="json")
    elif hasattr(value, "dict"):
        result = value.dict()
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise TypeError(f"cannot inspect Qdrant response: {type(value)!r}")
    if not isinstance(result, dict):
        raise TypeError("Qdrant response did not serialize to an object")
    return result


def _client(runtime: AppConfig) -> Any:
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=runtime.qdrant.url,
        api_key=runtime.qdrant.api_key(),
        prefer_grpc=runtime.qdrant.prefer_grpc,
        grpc_port=runtime.qdrant.grpc_port,
        timeout=runtime.qdrant.timeout_seconds,
    )


def _collection_schema(info: Any, dense_name: str) -> tuple[int, str, set[str]]:
    root = _model_dump(info)
    params = root.get("config", {}).get("params", {})
    vectors = params.get("vectors") or {}
    dense = vectors.get(dense_name)
    if not isinstance(dense, Mapping):
        raise ValueError(f"collection has no named dense vector {dense_name!r}")
    sparse = params.get("sparse_vectors") or {}
    return (
        int(dense.get("size", -1)),
        str(dense.get("distance", "")).casefold().removeprefix("distance."),
        {str(name) for name in sparse},
    )


def _ensure_collection(
    client: Any,
    runtime: AppConfig,
    index_config: IndexConfig,
    upload_config: UploadConfig,
    collection_name: str,
) -> bool:
    from qdrant_client import models

    created = False
    if not client.collection_exists(collection_name):
        optimizer = None
        if upload_config.disable_indexing_during_upload:
            optimizer = models.OptimizersConfigDiff(indexing_threshold=0)
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                runtime.qdrant.dense_vector_name: models.VectorParams(
                    size=runtime.dense.dimension,
                    distance=models.Distance.DOT,
                )
            },
            sparse_vectors_config={
                runtime.qdrant.sparse_vector_name: models.SparseVectorParams()
            },
            hnsw_config=models.HnswConfigDiff(
                m=index_config.qdrant_hnsw_m,
                ef_construct=index_config.qdrant_hnsw_ef_construction,
            ),
            optimizers_config=optimizer,
            on_disk_payload=True,
        )
        created = True
    info = client.get_collection(collection_name)
    dimension, distance, sparse_names = _collection_schema(
        info, runtime.qdrant.dense_vector_name
    )
    if dimension != runtime.dense.dimension:
        raise ValueError(
            f"Qdrant dense dimension is {dimension}; expected {runtime.dense.dimension}"
        )
    if distance != "dot":
        raise ValueError(f"Qdrant dense distance is {distance!r}; expected dot")
    if runtime.qdrant.sparse_vector_name not in sparse_names:
        raise ValueError(
            "target collection does not contain the configured sparse vector; "
            "use a new hybrid collection instead of a previous dense-only one"
        )
    payload_schema = _model_dump(info).get("payload_schema") or {}
    for field in ("document_id", "source_file"):
        if field not in payload_schema:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
    return created


def _checkpoint_value(
    *,
    collection_name: str,
    chunk_hash: str,
    chunk_count: int,
    runtime: AppConfig,
    sparse_hash: str,
    next_row: int,
    complete: bool,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "artifact_type": "qdrant_direct_ingestion_checkpoint",
        "collection_name": collection_name,
        "chunk_records_sha256": chunk_hash,
        "chunk_count": chunk_count,
        "dense": {
            "model_id": runtime.dense.model_id,
            "revision": runtime.dense.revision,
            "dimension": runtime.dense.dimension,
            "max_length": runtime.dense.max_length,
            "normalize_embeddings": runtime.dense.normalize_embeddings,
        },
        "vectors": {
            "dense": runtime.qdrant.dense_vector_name,
            "sparse": runtime.qdrant.sparse_vector_name,
        },
        "sparse_build_sha256": sparse_hash,
        "next_row": next_row,
        "complete": complete,
    }


def _resume_row(checkpoint_path: Path, expected: Mapping[str, Any]) -> int:
    if not checkpoint_path.is_file():
        return 0
    checkpoint = _load_json(checkpoint_path)
    for key in (
        "format_version",
        "collection_name",
        "chunk_records_sha256",
        "chunk_count",
        "dense",
        "vectors",
        "sparse_build_sha256",
    ):
        if checkpoint.get(key) != expected.get(key):
            raise ValueError(
                f"ingestion checkpoint mismatch for {key}; use a new state directory"
            )
    row = int(checkpoint.get("next_row", -1))
    if not 0 <= row <= int(expected["chunk_count"]):
        raise ValueError(f"invalid ingestion checkpoint row: {row}")
    return row


def _upsert(client: Any, collection: str, points: Sequence[Any], retries: int) -> None:
    for attempt in range(retries + 1):
        try:
            client.upsert(collection_name=collection, points=points, wait=True)
            return
        except Exception:
            if attempt >= retries:
                raise
            delay = min(2 ** (attempt + 1), 30)
            print(f"Qdrant upsert failed; retry in {delay}s", flush=True)
            time.sleep(delay)


def _wait_green(client: Any, collection: str, timeout_seconds: int) -> str:
    if timeout_seconds <= 0:
        return "not_waited"
    deadline = time.monotonic() + timeout_seconds
    while True:
        info = client.get_collection(collection)
        raw = getattr(info, "status", "unknown")
        status = str(getattr(raw, "value", raw)).casefold()
        if status == "green":
            return status
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Qdrant collection did not become green; last status={status}"
            )
        time.sleep(5)


def _validate_samples(
    client: Any,
    collection: str,
    chunks_path: Path,
    chunk_count: int,
) -> list[dict[str, Any]]:
    rows = {0, chunk_count // 2, chunk_count - 1}
    expected: dict[str, tuple[int, str]] = {}
    for row, record in enumerate(iter_chunk_records(chunks_path)):
        if row in rows:
            normalized = normalize_chunk_record(record)
            expected[point_id_for_chunk(normalized["chunk_id"])] = (
                row,
                normalized["chunk_id"],
            )
    points = client.retrieve(
        collection_name=collection,
        ids=list(expected),
        with_payload=True,
        with_vectors=False,
    )
    found: list[dict[str, Any]] = []
    for point in points:
        point_id = str(getattr(point, "id"))
        payload = getattr(point, "payload", None)
        if point_id not in expected or not isinstance(payload, Mapping):
            raise ValueError("unexpected Qdrant sample response")
        row, chunk_id = expected[point_id]
        if payload.get("chunk_id") != chunk_id or int(payload.get("vector_row", -1)) != row:
            raise ValueError(f"Qdrant sample payload mismatch at row {row}")
        found.append({"row": row, "chunk_id": chunk_id, "point_id": point_id})
    if len(found) != len(expected):
        raise ValueError("Qdrant sample validation did not retrieve all expected points")
    return sorted(found, key=lambda item: int(item["row"]))


def ingest_direct(
    chunks_path: str | Path,
    state_dir: str | Path,
    runtime: AppConfig,
    index_config: IndexConfig,
    upload_config: UploadConfig,
    *,
    collection_name: str | None = None,
    rebuild_sparse: bool = False,
) -> Path:
    """Encode long chunks and upsert named dense+sparse vectors directly.

    No FAISS index is created.  Point IDs are deterministic, and the local
    checkpoint is written after every acknowledged batch so interrupted AWS
    uploads can safely resume.
    """

    from qdrant_client import models

    source = Path(chunks_path).expanduser().resolve()
    state_root = Path(state_dir).expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    resolved_collection = collection_name or runtime.qdrant.collection
    chunk_hash, chunk_count = canonical_chunk_hash(source)
    if chunk_count <= 0:
        raise ValueError("cannot ingest an empty chunk corpus")

    sparse = prepare_sparse_corpus(
        source,
        state_root / "bm25_build",
        index_config,
        overwrite=rebuild_sparse,
    )
    sparse_matrix = sparse.load_csr()
    sparse_manifest_hash = sha256_file(sparse.directory / "manifest.json")
    query_manifest = export_query_bundle(
        sparse,
        state_root / "sparse_query",
        collection_name=resolved_collection,
        vector_name=runtime.qdrant.sparse_vector_name,
    )
    expected = _checkpoint_value(
        collection_name=resolved_collection,
        chunk_hash=chunk_hash,
        chunk_count=chunk_count,
        runtime=runtime,
        sparse_hash=sparse_manifest_hash,
        next_row=0,
        complete=False,
    )
    checkpoint_path = state_root / "checkpoint.json"
    next_row = _resume_row(checkpoint_path, expected)
    client = _client(runtime)
    created = _ensure_collection(
        client,
        runtime,
        index_config,
        upload_config,
        resolved_collection,
    )
    existing_count = int(
        client.count(collection_name=resolved_collection, exact=True).count
    )
    if not checkpoint_path.is_file() and existing_count:
        raise ValueError(
            f"target collection contains {existing_count} points but this job has no "
            "checkpoint; use a new collection or the matching state directory"
        )
    if existing_count < next_row or existing_count > min(
        next_row + upload_config.batch_size, chunk_count
    ):
        raise ValueError(
            f"Qdrant point count {existing_count} is incompatible with checkpoint "
            f"row {next_row}"
        )
    if not checkpoint_path.is_file():
        _write_json(checkpoint_path, expected)

    encoder = DenseEncoder(runtime.dense) if next_row < chunk_count else None
    records = itertools.islice(iter_chunk_records(source), next_row, None)
    row = next_row
    started = time.monotonic()
    while row < chunk_count:
        batch = list(itertools.islice(records, upload_config.batch_size))
        if not batch:
            break
        normalized = [normalize_chunk_record(record) for record in batch]
        assert encoder is not None
        dense_vectors = encoder.encode_documents(
            [record["retrieval_text"] for record in normalized],
            show_progress=False,
        )
        points = []
        for offset, (raw_record, record) in enumerate(zip(batch, normalized)):
            vector_row = row + offset
            start = int(sparse_matrix.indptr[vector_row])
            end = int(sparse_matrix.indptr[vector_row + 1])
            sparse_indices = sparse_matrix.indices[start:end]
            sparse_values = sparse_matrix.data[start:end]
            payload = build_payload(raw_record, vector_row=vector_row)
            points.append(
                models.PointStruct(
                    id=point_id_for_chunk(record["chunk_id"]),
                    vector={
                        runtime.qdrant.dense_vector_name: dense_vectors[offset].tolist(),
                        runtime.qdrant.sparse_vector_name: models.SparseVector(
                            indices=[int(value) for value in sparse_indices],
                            values=[float(value) for value in sparse_values],
                        ),
                    },
                    payload=payload,
                )
            )
        _upsert(
            client,
            resolved_collection,
            points,
            upload_config.max_retries,
        )
        row += len(batch)
        _write_json(
            checkpoint_path,
            _checkpoint_value(
                collection_name=resolved_collection,
                chunk_hash=chunk_hash,
                chunk_count=chunk_count,
                runtime=runtime,
                sparse_hash=sparse_manifest_hash,
                next_row=row,
                complete=False,
            ),
        )
        elapsed = max(time.monotonic() - started, 1e-9)
        rate = (row - next_row) / elapsed
        print(
            f"Uploaded {row}/{chunk_count} ({100 * row / chunk_count:.2f}%) | "
            f"{rate:.1f} chunks/s",
            flush=True,
        )
    if row != chunk_count:
        raise RuntimeError(f"chunk stream ended at {row}; expected {chunk_count}")
    final_count = int(client.count(collection_name=resolved_collection, exact=True).count)
    if final_count != chunk_count:
        raise ValueError(
            f"Qdrant contains {final_count} points; expected exactly {chunk_count}"
        )
    if upload_config.disable_indexing_during_upload:
        client.update_collection(
            collection_name=resolved_collection,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=upload_config.optimizer_indexing_threshold_kb
            ),
        )
    status = _wait_green(
        client, resolved_collection, upload_config.optimizer_timeout_seconds
    )
    samples = _validate_samples(
        client, resolved_collection, source, chunk_count
    )
    manifest_path = state_root / "ingestion_manifest.json"
    manifest = {
        **_checkpoint_value(
            collection_name=resolved_collection,
            chunk_hash=chunk_hash,
            chunk_count=chunk_count,
            runtime=runtime,
            sparse_hash=sparse_manifest_hash,
            next_row=chunk_count,
            complete=True,
        ),
        "artifact_type": "qdrant_hybrid_direct_ingestion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qdrant_url": runtime.qdrant.url,
        "collection_created_by_job": created,
        "optimizer_status": status,
        "payload_schema": PAYLOAD_SCHEMA,
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "sparse_query_manifest": str(query_manifest),
        "sample_validation": samples,
    }
    _write_json(manifest_path, manifest)
    _write_json(checkpoint_path, manifest)
    return manifest_path
