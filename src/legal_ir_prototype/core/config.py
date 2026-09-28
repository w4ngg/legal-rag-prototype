from __future__ import annotations

"""Typed application configuration and YAML loading."""

import os
import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _section(root: Mapping[str, Any], name: str) -> dict[str, Any]:
    if name not in root:
        raise ValueError(f"missing configuration section: {name}")
    return _mapping(root[name], name)


def _reject_unknown(values: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} configuration keys: {unknown}")


def _text(value: Any, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} is required")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


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


def _revision(value: Any, name: str) -> str:
    revision = _text(value, name)
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError(f"{name} must be a pinned 40-character commit SHA")
    return revision.lower()


def _non_negative(value: Any, name: str) -> float:
    result = float(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _optional_finite_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite or null")
    return result


def _runtime_fields(values: Mapping[str, Any], name: str) -> tuple[str, str]:
    device = _text(values.get("device"), f"{name}.device")
    dtype = _text(values.get("dtype"), f"{name}.dtype")
    if device not in {"auto", "cpu", "cuda", "mps"} and not device.startswith(
        "cuda:"
    ):
        raise ValueError(f"unsupported {name}.device: {device}")
    if dtype not in {"auto", "float16", "bfloat16", "float32"}:
        raise ValueError(f"unsupported {name}.dtype: {dtype}")
    return device, dtype


@dataclass(frozen=True, slots=True)
class DenseConfig:
    model_id: str
    revision: str
    max_length: int
    dimension: int
    batch_size: int
    normalize_embeddings: bool
    device: str
    dtype: str
    trust_remote_code: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DenseConfig":
        values = _mapping(raw, "dense")
        _reject_unknown(
            values,
            {
                "model_id",
                "revision",
                "max_length",
                "dimension",
                "batch_size",
                "normalize_embeddings",
                "device",
                "dtype",
                "trust_remote_code",
            },
            "dense",
        )
        device, dtype = _runtime_fields(values, "dense")
        return cls(
            model_id=_text(values.get("model_id"), "dense.model_id"),
            revision=_revision(values.get("revision"), "dense.revision"),
            max_length=_positive_int(values.get("max_length"), "dense.max_length"),
            dimension=_positive_int(values.get("dimension"), "dense.dimension"),
            batch_size=_positive_int(values.get("batch_size"), "dense.batch_size"),
            normalize_embeddings=bool(values.get("normalize_embeddings")),
            device=device,
            dtype=dtype,
            trust_remote_code=bool(values.get("trust_remote_code", False)),
        )


@dataclass(frozen=True, slots=True)
class SparseConfig:
    query_encoder_manifest: Path

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        config_dir: Path,
    ) -> "SparseConfig":
        values = _mapping(raw, "sparse")
        _reject_unknown(values, {"query_encoder_manifest"}, "sparse")
        path = Path(
            _text(
                values.get("query_encoder_manifest"),
                "sparse.query_encoder_manifest",
            )
        ).expanduser()
        if not path.is_absolute():
            path = config_dir / path
        return cls(query_encoder_manifest=path.resolve())


@dataclass(frozen=True, slots=True)
class QdrantConfig:
    url: str
    api_key_env: str | None
    collection: str
    dense_vector_name: str
    sparse_vector_name: str
    prefer_grpc: bool
    grpc_port: int
    timeout_seconds: float
    exact: bool
    hnsw_ef: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "QdrantConfig":
        values = _mapping(raw, "qdrant")
        _reject_unknown(
            values,
            {
                "url",
                "api_key_env",
                "collection",
                "dense_vector_name",
                "sparse_vector_name",
                "prefer_grpc",
                "grpc_port",
                "timeout_seconds",
                "exact",
                "hnsw_ef",
            },
            "qdrant",
        )
        dense_name = _text(
            values.get("dense_vector_name"), "qdrant.dense_vector_name"
        )
        sparse_name = _text(
            values.get("sparse_vector_name"), "qdrant.sparse_vector_name"
        )
        if dense_name == sparse_name:
            raise ValueError("dense and sparse Qdrant vector names must differ")
        timeout = float(values.get("timeout_seconds"))
        if timeout <= 0:
            raise ValueError("qdrant.timeout_seconds must be positive")
        api_key_env = values.get("api_key_env")
        return cls(
            url=_text(values.get("url"), "qdrant.url"),
            api_key_env=(
                None
                if api_key_env in {None, ""}
                else _text(api_key_env, "qdrant.api_key_env")
            ),
            collection=_text(values.get("collection"), "qdrant.collection"),
            dense_vector_name=dense_name,
            sparse_vector_name=sparse_name,
            prefer_grpc=bool(values.get("prefer_grpc", False)),
            grpc_port=_positive_int(values.get("grpc_port"), "qdrant.grpc_port"),
            timeout_seconds=timeout,
            exact=bool(values.get("exact", True)),
            hnsw_ef=_positive_int(values.get("hnsw_ef"), "qdrant.hnsw_ef"),
        )

    def api_key(self) -> str | None:
        if self.api_key_env is None:
            return None
        value = os.environ.get(self.api_key_env)
        return value if value else None


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    dense_top_k: int
    sparse_top_k: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RetrievalConfig":
        values = _mapping(raw, "retrieval")
        _reject_unknown(
            values,
            {"dense_top_k", "sparse_top_k"},
            "retrieval",
        )
        return cls(
            dense_top_k=_positive_int(
                values.get("dense_top_k"), "retrieval.dense_top_k"
            ),
            sparse_top_k=_positive_int(
                values.get("sparse_top_k"), "retrieval.sparse_top_k"
            ),
        )


@dataclass(frozen=True, slots=True)
class FusionConfig:
    algorithm: str
    rrf_k: int
    dense_weight: float
    sparse_weight: float
    candidate_documents: int | None
    evidence_chunks_per_document: int | None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FusionConfig":
        values = _mapping(raw, "fusion")
        _reject_unknown(
            values,
            {
                "algorithm",
                "rrf_k",
                "dense_weight",
                "sparse_weight",
                "candidate_documents",
                "evidence_chunks_per_document",
            },
            "fusion",
        )
        algorithm = _text(values.get("algorithm"), "fusion.algorithm")
        if algorithm != "weighted_rrf":
            raise ValueError(f"unsupported fusion.algorithm: {algorithm}")
        dense_weight = _non_negative(
            values.get("dense_weight"), "fusion.dense_weight"
        )
        sparse_weight = _non_negative(
            values.get("sparse_weight"), "fusion.sparse_weight"
        )
        if dense_weight + sparse_weight <= 0:
            raise ValueError("at least one fusion channel weight must be positive")
        candidate_documents = values.get("candidate_documents")
        evidence_chunks = values.get("evidence_chunks_per_document")
        return cls(
            algorithm=algorithm,
            rrf_k=_positive_int(values.get("rrf_k"), "fusion.rrf_k"),
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            candidate_documents=(
                None
                if candidate_documents is None
                else _positive_int(
                    candidate_documents, "fusion.candidate_documents"
                )
            ),
            evidence_chunks_per_document=(
                None
                if evidence_chunks is None
                else _positive_int(
                    evidence_chunks,
                    "fusion.evidence_chunks_per_document",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    model_id: str
    revision: str
    max_length: int
    batch_size: int
    output_top_k: int
    device: str
    dtype: str
    trust_remote_code: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RerankerConfig":
        values = _mapping(raw, "reranker")
        _reject_unknown(
            values,
            {
                "model_id",
                "revision",
                "max_length",
                "batch_size",
                "output_top_k",
                "device",
                "dtype",
                "trust_remote_code",
            },
            "reranker",
        )
        device, dtype = _runtime_fields(values, "reranker")
        return cls(
            model_id=_text(values.get("model_id"), "reranker.model_id"),
            revision=_revision(values.get("revision"), "reranker.revision"),
            max_length=_positive_int(
                values.get("max_length"), "reranker.max_length"
            ),
            batch_size=_positive_int(
                values.get("batch_size"), "reranker.batch_size"
            ),
            output_top_k=_positive_int(
                values.get("output_top_k"), "reranker.output_top_k"
            ),
            device=device,
            dtype=dtype,
            trust_remote_code=bool(values.get("trust_remote_code", False)),
        )


@dataclass(frozen=True, slots=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str | None
    api_key_env: str | None
    timeout_seconds: float
    max_retries: int
    temperature: float
    max_output_tokens: int
    structured_output_mode: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LLMConfig":
        values = _mapping(raw, "llm")
        _reject_unknown(
            values,
            {
                "provider",
                "model",
                "base_url",
                "api_key_env",
                "timeout_seconds",
                "max_retries",
                "temperature",
                "max_output_tokens",
                "structured_output_mode",
            },
            "llm",
        )
        provider = _text(values.get("provider"), "llm.provider")
        if provider not in {"openai_compatible", "fake"}:
            raise ValueError(f"unsupported llm.provider: {provider}")
        base_url_value = values.get("base_url")
        base_url = (
            None
            if base_url_value in {None, ""}
            else _text(base_url_value, "llm.base_url").rstrip("/")
        )
        if provider == "openai_compatible" and base_url is None:
            raise ValueError("llm.base_url is required for openai_compatible")
        api_key_env_value = values.get("api_key_env")
        api_key_env = (
            None
            if api_key_env_value in {None, ""}
            else _text(api_key_env_value, "llm.api_key_env")
        )
        timeout_seconds = float(values.get("timeout_seconds"))
        if timeout_seconds <= 0:
            raise ValueError("llm.timeout_seconds must be positive")
        temperature = float(values.get("temperature"))
        if temperature < 0:
            raise ValueError("llm.temperature must be non-negative")
        structured_output_mode = _text(
            values.get("structured_output_mode"),
            "llm.structured_output_mode",
        )
        if structured_output_mode not in {"json_schema", "json_object"}:
            raise ValueError(
                "llm.structured_output_mode must be json_schema or json_object"
            )
        return cls(
            provider=provider,
            model=_text(values.get("model"), "llm.model"),
            base_url=base_url,
            api_key_env=api_key_env,
            timeout_seconds=timeout_seconds,
            max_retries=_non_negative_int(
                values.get("max_retries"),
                "llm.max_retries",
            ),
            temperature=temperature,
            max_output_tokens=_positive_int(
                values.get("max_output_tokens"),
                "llm.max_output_tokens",
            ),
            structured_output_mode=structured_output_mode,
        )

    def api_key(self) -> str | None:
        if self.api_key_env is None:
            return None
        value = os.environ.get(self.api_key_env)
        return value if value else None


@dataclass(frozen=True, slots=True)
class EvidenceConfig:
    min_sources: int = 1
    min_top_reranker_score: float | None = None
    min_top_score_margin: float | None = None
    abstention_answer: str = (
        "Không đủ bằng chứng trong các nguồn pháp luật được truy xuất để trả lời."
    )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EvidenceConfig":
        values = _mapping(raw, "rag.evidence")
        _reject_unknown(
            values,
            {
                "min_sources",
                "min_top_reranker_score",
                "min_top_score_margin",
                "abstention_answer",
            },
            "rag.evidence",
        )
        margin = _optional_finite_float(
            values.get("min_top_score_margin"),
            "rag.evidence.min_top_score_margin",
        )
        if margin is not None and margin < 0:
            raise ValueError(
                "rag.evidence.min_top_score_margin must be non-negative"
            )
        default = cls()
        return cls(
            min_sources=_positive_int(
                values.get("min_sources", default.min_sources),
                "rag.evidence.min_sources",
            ),
            min_top_reranker_score=_optional_finite_float(
                values.get("min_top_reranker_score"),
                "rag.evidence.min_top_reranker_score",
            ),
            min_top_score_margin=margin,
            abstention_answer=_text(
                values.get("abstention_answer", default.abstention_answer),
                "rag.evidence.abstention_answer",
            ),
        )


@dataclass(frozen=True, slots=True)
class RAGConfig:
    max_context_tokens: int = 10_000
    max_chunks: int | None = 8
    max_chunks_per_document: int | None = 2
    include_legal_metadata: bool = True
    merge_adjacent_chunks: bool = False
    evidence: EvidenceConfig = EvidenceConfig()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RAGConfig":
        values = _mapping(raw, "rag")
        _reject_unknown(
            values,
            {
                "max_context_tokens",
                "max_chunks",
                "max_chunks_per_document",
                "include_legal_metadata",
                "merge_adjacent_chunks",
                "evidence",
            },
            "rag",
        )
        default = cls()
        max_chunks = values.get("max_chunks", default.max_chunks)
        max_chunks_per_document = values.get(
            "max_chunks_per_document",
            default.max_chunks_per_document,
        )
        return cls(
            max_context_tokens=_positive_int(
                values.get("max_context_tokens", default.max_context_tokens),
                "rag.max_context_tokens",
            ),
            max_chunks=(
                None
                if max_chunks is None
                else _positive_int(max_chunks, "rag.max_chunks")
            ),
            max_chunks_per_document=(
                None
                if max_chunks_per_document is None
                else _positive_int(
                    max_chunks_per_document,
                    "rag.max_chunks_per_document",
                )
            ),
            include_legal_metadata=bool(
                values.get(
                    "include_legal_metadata",
                    default.include_legal_metadata,
                )
            ),
            merge_adjacent_chunks=bool(
                values.get(
                    "merge_adjacent_chunks",
                    default.merge_adjacent_chunks,
                )
            ),
            evidence=(
                default.evidence
                if "evidence" not in values
                else EvidenceConfig.from_mapping(values["evidence"])
            ),
        )


@dataclass(frozen=True, slots=True)
class AppConfig:
    dense: DenseConfig
    sparse: SparseConfig
    qdrant: QdrantConfig
    retrieval: RetrievalConfig
    fusion: FusionConfig
    reranker: RerankerConfig
    source_path: Path
    llm: LLMConfig | None = None
    rag: RAGConfig = RAGConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        root = _mapping(raw, "configuration root")
        required = {"dense", "sparse", "qdrant", "retrieval", "fusion", "reranker"}
        allowed = required | {"llm", "rag"}
        _reject_unknown(root, allowed, "root")
        missing = sorted(required - set(root))
        if missing:
            raise ValueError(f"missing configuration sections: {missing}")
        retrieval = RetrievalConfig.from_mapping(_section(root, "retrieval"))
        reranker = RerankerConfig.from_mapping(_section(root, "reranker"))
        return cls(
            dense=DenseConfig.from_mapping(_section(root, "dense")),
            sparse=SparseConfig.from_mapping(
                _section(root, "sparse"), config_dir=source.parent
            ),
            qdrant=QdrantConfig.from_mapping(_section(root, "qdrant")),
            retrieval=retrieval,
            fusion=FusionConfig.from_mapping(_section(root, "fusion")),
            reranker=reranker,
            source_path=source,
            llm=(
                None
                if "llm" not in root
                else LLMConfig.from_mapping(_section(root, "llm"))
            ),
            rag=(
                RAGConfig()
                if "rag" not in root
                else RAGConfig.from_mapping(_section(root, "rag"))
            ),
        )
