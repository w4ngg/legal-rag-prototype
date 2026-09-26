from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..core import AppConfig


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _reject_unknown(values: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} keys: {unknown}")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    input_pattern: str
    short_max_characters: int
    long_max_characters: int
    long_overlap_short_chunks: int
    include_document_title: bool
    title_max_characters: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ChunkingConfig":
        values = _mapping(raw, "chunking")
        allowed = {
            "input_pattern",
            "short_max_characters",
            "long_max_characters",
            "long_overlap_short_chunks",
            "include_document_title",
            "title_max_characters",
        }
        _reject_unknown(values, allowed, "chunking")
        input_pattern = str(values.get("input_pattern", "context_*.json")).strip()
        if not input_pattern:
            raise ValueError("chunking.input_pattern must not be empty")
        short_max = _positive_int(
            values.get("short_max_characters"),
            "chunking.short_max_characters",
        )
        long_max = _positive_int(
            values.get("long_max_characters"),
            "chunking.long_max_characters",
        )
        if long_max < short_max:
            raise ValueError(
                "chunking.long_max_characters must be >= short_max_characters"
            )
        return cls(
            input_pattern=input_pattern,
            short_max_characters=short_max,
            long_max_characters=long_max,
            long_overlap_short_chunks=_non_negative_int(
                values.get("long_overlap_short_chunks"),
                "chunking.long_overlap_short_chunks",
            ),
            include_document_title=bool(
                values.get("include_document_title", True)
            ),
            title_max_characters=_positive_int(
                values.get("title_max_characters", 300),
                "chunking.title_max_characters",
            ),
        )


@dataclass(frozen=True, slots=True)
class IndexConfig:
    bm25_method: str
    bm25_k1: float
    bm25_b: float
    qdrant_hnsw_m: int
    qdrant_hnsw_ef_construction: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "IndexConfig":
        values = _mapping(raw, "index")
        allowed = {
            "bm25_method",
            "bm25_k1",
            "bm25_b",
            "qdrant_hnsw_m",
            "qdrant_hnsw_ef_construction",
        }
        _reject_unknown(values, allowed, "index")
        method = str(values.get("bm25_method", "lucene")).strip()
        if not method:
            raise ValueError("index.bm25_method must not be empty")
        k1 = float(values.get("bm25_k1", 1.2))
        b = float(values.get("bm25_b", 0.75))
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("invalid BM25 k1/b")
        return cls(
            bm25_method=method,
            bm25_k1=k1,
            bm25_b=b,
            qdrant_hnsw_m=_positive_int(
                values.get("qdrant_hnsw_m", 32), "index.qdrant_hnsw_m"
            ),
            qdrant_hnsw_ef_construction=_positive_int(
                values.get("qdrant_hnsw_ef_construction", 200),
                "index.qdrant_hnsw_ef_construction",
            ),
        )


@dataclass(frozen=True, slots=True)
class UploadConfig:
    batch_size: int
    max_retries: int
    disable_indexing_during_upload: bool
    optimizer_indexing_threshold_kb: int
    optimizer_timeout_seconds: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "UploadConfig":
        values = _mapping(raw, "upload")
        allowed = {
            "batch_size",
            "max_retries",
            "disable_indexing_during_upload",
            "optimizer_indexing_threshold_kb",
            "optimizer_timeout_seconds",
        }
        _reject_unknown(values, allowed, "upload")
        return cls(
            batch_size=_positive_int(values.get("batch_size", 256), "upload.batch_size"),
            max_retries=_non_negative_int(
                values.get("max_retries", 5), "upload.max_retries"
            ),
            disable_indexing_during_upload=bool(
                values.get("disable_indexing_during_upload", True)
            ),
            optimizer_indexing_threshold_kb=_positive_int(
                values.get("optimizer_indexing_threshold_kb", 10000),
                "upload.optimizer_indexing_threshold_kb",
            ),
            optimizer_timeout_seconds=_non_negative_int(
                values.get("optimizer_timeout_seconds", 3600),
                "upload.optimizer_timeout_seconds",
            ),
        )


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    runtime_config_path: Path
    runtime: AppConfig
    chunking: ChunkingConfig
    index: IndexConfig
    upload: UploadConfig
    source_path: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IngestionConfig":
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        root = _mapping(raw, "ingestion configuration")
        allowed = {"runtime_config", "chunking", "index", "upload"}
        _reject_unknown(root, allowed, "ingestion configuration")
        missing = sorted(allowed - set(root))
        if missing:
            raise ValueError(f"missing ingestion sections: {missing}")
        runtime_path = Path(str(root["runtime_config"])).expanduser()
        if not runtime_path.is_absolute():
            runtime_path = source.parent / runtime_path
        runtime_path = runtime_path.resolve()
        return cls(
            runtime_config_path=runtime_path,
            runtime=AppConfig.from_yaml(runtime_path),
            chunking=ChunkingConfig.from_mapping(root["chunking"]),
            index=IndexConfig.from_mapping(root["index"]),
            upload=UploadConfig.from_mapping(root["upload"]),
            source_path=source,
        )
