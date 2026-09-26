from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote
from urllib.request import Request, urlopen

from ..core import AppConfig


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        result = value.model_dump(mode="json")
    elif hasattr(value, "dict"):
        result = value.dict()
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise TypeError(f"cannot serialize response type: {type(value)!r}")
    if not isinstance(result, dict):
        raise TypeError("response did not serialize to an object")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _copy_sparse_query_bundle(
    runtime: AppConfig,
    destination: Path,
    *,
    collection_name: str,
) -> dict[str, Any]:
    source_manifest = runtime.sparse.query_encoder_manifest
    if not source_manifest.is_file():
        raise FileNotFoundError(
            "sparse query manifest is required beside a Qdrant snapshot: "
            f"{source_manifest}"
        )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("sparse query manifest must contain an object")
    if manifest.get("artifact_type") != "bm25s_sparse_query_encoder":
        raise ValueError("unsupported sparse query bundle")
    manifest_collection = str(manifest.get("collection_name") or "")
    if manifest_collection and manifest_collection != collection_name:
        raise ValueError(
            "sparse query bundle belongs to a different collection: "
            f"{manifest_collection!r} != {collection_name!r}"
        )
    filenames = {
        source_manifest.name,
        str((manifest.get("vocabulary") or {}).get("filename") or ""),
        str((manifest.get("bm25") or {}).get("filename") or ""),
    }
    if "" in filenames:
        raise ValueError("sparse query manifest has incomplete artifact filenames")
    bundle_dir = destination / "sparse_query"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for filename in sorted(filenames):
        safe_name = Path(filename).name
        if safe_name != filename:
            raise ValueError(f"unsafe sparse artifact filename: {filename!r}")
        source = source_manifest.parent / safe_name
        if not source.is_file():
            raise FileNotFoundError(f"missing sparse query artifact: {source}")
        target = bundle_dir / safe_name
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        artifacts[safe_name] = _sha256(target)
    return {
        "directory": bundle_dir.name,
        "manifest": source_manifest.name,
        "artifacts_sha256": artifacts,
    }


def _client(runtime: AppConfig) -> Any:
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=runtime.qdrant.url,
        api_key=runtime.qdrant.api_key(),
        prefer_grpc=runtime.qdrant.prefer_grpc,
        grpc_port=runtime.qdrant.grpc_port,
        timeout=runtime.qdrant.timeout_seconds,
    )


def create_local_snapshot_bundle(
    runtime: AppConfig,
    output_dir: str | Path,
    *,
    collection_name: str | None = None,
) -> Path:
    """Create and download a local Qdrant snapshot plus migration metadata."""

    collection = collection_name or runtime.qdrant.collection
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    client = _client(runtime)
    if not client.collection_exists(collection):
        raise ValueError(f"Qdrant collection does not exist: {collection}")
    info = client.get_collection(collection)
    snapshot = client.create_snapshot(collection_name=collection, wait=True)
    snapshot_data = _model_dump(snapshot)
    snapshot_name = str(snapshot_data.get("name") or "").strip()
    if not snapshot_name:
        raise ValueError("Qdrant did not return a snapshot filename")
    target = destination / snapshot_name
    temporary = target.with_name(f".{target.name}.{os.getpid()}.part")
    endpoint = (
        runtime.qdrant.url.rstrip("/")
        + "/collections/"
        + quote(collection, safe="")
        + "/snapshots/"
        + quote(snapshot_name, safe="")
    )
    headers: dict[str, str] = {}
    api_key = runtime.qdrant.api_key()
    if api_key:
        headers["api-key"] = api_key
    request = Request(endpoint, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=runtime.qdrant.timeout_seconds) as response:
            with temporary.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    count = int(client.count(collection_name=collection, exact=True).count)
    server = _model_dump(client.info())
    sparse_bundle = _copy_sparse_query_bundle(
        runtime,
        destination,
        collection_name=collection,
    )
    manifest_path = destination / f"{snapshot_name}.manifest.json"
    _write_json(
        manifest_path,
        {
            "format_version": 1,
            "artifact_type": "qdrant_collection_snapshot_migration",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "qdrant_url": runtime.qdrant.url,
                "qdrant_version": server.get("version"),
                "collection": collection,
                "point_count": count,
                "collection_info": _model_dump(info),
            },
            "snapshot": {
                "filename": target.name,
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
                "qdrant_checksum": snapshot_data.get("checksum"),
                "qdrant_metadata": snapshot_data,
            },
            "runtime_contract": {
                "dense_model_id": runtime.dense.model_id,
                "dense_revision": runtime.dense.revision,
                "dense_dimension": runtime.dense.dimension,
                "dense_vector_name": runtime.qdrant.dense_vector_name,
                "sparse_vector_name": runtime.qdrant.sparse_vector_name,
                "sparse_query_manifest": str(
                    runtime.sparse.query_encoder_manifest
                ),
            },
            "sparse_query_bundle": sparse_bundle,
        },
    )
    return manifest_path


def restore_remote_snapshot(
    runtime: AppConfig,
    *,
    snapshot_url: str,
    target_collection: str,
    checksum: str | None = None,
) -> dict[str, Any]:
    """Ask the target Qdrant server to restore a snapshot from an HTTPS URL.

    For AWS the URL is normally a short-lived presigned S3 URL reachable by
    the Qdrant EC2 instance.  The target collection must not already exist.
    """

    from qdrant_client import models

    url = snapshot_url.strip()
    collection = target_collection.strip()
    if not url.startswith(("https://", "http://")):
        raise ValueError("snapshot_url must be an HTTP(S) URL visible to Qdrant")
    if not collection:
        raise ValueError("target_collection must not be empty")
    client = _client(runtime)
    if client.collection_exists(collection):
        raise FileExistsError(
            f"target collection already exists: {collection}; refusing overwrite"
        )
    client.recover_snapshot(
        collection_name=collection,
        location=url,
        checksum=checksum,
        priority=models.SnapshotPriority.SNAPSHOT,
        wait=True,
    )
    info = client.get_collection(collection)
    data = _model_dump(info)
    params = data.get("config", {}).get("params", {})
    vectors = params.get("vectors") or {}
    dense = vectors.get(runtime.qdrant.dense_vector_name)
    sparse = params.get("sparse_vectors") or {}
    if not isinstance(dense, Mapping):
        raise ValueError("restored collection lacks configured dense vector")
    if int(dense.get("size", -1)) != runtime.dense.dimension:
        raise ValueError("restored collection has unexpected dense dimension")
    if runtime.qdrant.sparse_vector_name not in sparse:
        raise ValueError("restored collection lacks configured sparse vector")
    count = int(client.count(collection_name=collection, exact=True).count)
    return {
        "collection": collection,
        "point_count": count,
        "status": str(getattr(getattr(info, "status", None), "value", "unknown")),
        "dense_vector_name": runtime.qdrant.dense_vector_name,
        "sparse_vector_name": runtime.qdrant.sparse_vector_name,
    }
