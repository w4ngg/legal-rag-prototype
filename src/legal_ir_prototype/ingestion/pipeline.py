from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .chunking import build_long_chunks
from .config import IngestionConfig
from .qdrant import ingest_direct
from .schema import canonical_chunk_hash
from .sparse_index import prepare_sparse_corpus


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


def prepare_pipeline(
    raw_dir: str | Path,
    work_dir: str | Path,
    config: IngestionConfig,
    *,
    collection_name: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Create long chunks and the corpus-global BM25 ingestion state."""

    root = Path(work_dir).expanduser().resolve()
    chunk_dir = root / "chunks"
    chunks_path = chunk_dir / "long_chunks.jsonl"
    chunk_manifest = chunk_dir / "manifest.json"
    if overwrite or not (chunks_path.is_file() and chunk_manifest.is_file()):
        if chunks_path.exists() != chunk_manifest.exists() and not overwrite:
            raise ValueError(
                "partial chunk artifacts exist; inspect them and rerun with --overwrite"
            )
        chunks_path = build_long_chunks(
            raw_dir,
            chunk_dir,
            config.chunking,
            overwrite=overwrite,
        )
    collection = collection_name or config.runtime.qdrant.collection
    state_dir = root / "qdrant" / collection
    sparse = prepare_sparse_corpus(
        chunks_path,
        state_dir / "bm25_build",
        config.index,
        overwrite=overwrite,
    )
    chunk_hash, chunk_count = canonical_chunk_hash(chunks_path)
    manifest_path = root / "prepare_manifest.json"
    _write_json(
        manifest_path,
        {
            "format_version": 1,
            "artifact_type": "legal_ir_direct_ingestion_preparation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_dir": str(Path(raw_dir).expanduser().resolve()),
            "chunks_path": str(chunks_path),
            "chunk_records_sha256": chunk_hash,
            "chunk_count": chunk_count,
            "bm25_build_state": str(sparse.directory),
            "target_qdrant": {
                "url": config.runtime.qdrant.url,
                "collection": collection,
                "dense_vector": config.runtime.qdrant.dense_vector_name,
                "sparse_vector": config.runtime.qdrant.sparse_vector_name,
            },
        },
    )
    return manifest_path


def run_pipeline(
    raw_dir: str | Path,
    work_dir: str | Path,
    config: IngestionConfig,
    *,
    collection_name: str | None = None,
    overwrite_preparation: bool = False,
) -> Path:
    root = Path(work_dir).expanduser().resolve()
    prepare_pipeline(
        raw_dir,
        root,
        config,
        collection_name=collection_name,
        overwrite=overwrite_preparation,
    )
    chunks_path = root / "chunks" / "long_chunks.jsonl"
    collection = collection_name or config.runtime.qdrant.collection
    return ingest_direct(
        chunks_path,
        root / "qdrant" / collection,
        config.runtime,
        config.index,
        config.upload,
        collection_name=collection,
        # prepare_pipeline already built or validated this exact state.
        rebuild_sparse=False,
    )
