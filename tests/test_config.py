from __future__ import annotations

from pathlib import Path

import pytest

from legal_ir_prototype.core import AppConfig


CONFIG = """
dense:
  model_id: org/embedding
  revision: 1111111111111111111111111111111111111111
  max_length: 2048
  dimension: 1024
  batch_size: 8
  normalize_embeddings: true
  device: cpu
  dtype: float32
  trust_remote_code: false
sparse:
  query_encoder_manifest: assets/sparse/manifest.json
qdrant:
  url: http://localhost:6333
  api_key_env: QDRANT_API_KEY
  collection: legal-long
  dense_vector_name: dense
  sparse_vector_name: sparse
  prefer_grpc: false
  grpc_port: 6334
  timeout_seconds: 30
  exact: true
  hnsw_ef: 128
retrieval:
  dense_top_k: 80
  sparse_top_k: 40
fusion:
  algorithm: weighted_rrf
  rrf_k: 60
  dense_weight: 1.0
  sparse_weight: 1.0
  candidate_documents: 120
  evidence_chunks_per_document: 120
reranker:
  model_id: org/reranker
  revision: 2222222222222222222222222222222222222222
  max_length: 2304
  batch_size: 4
  output_top_k: 20
  device: cpu
  dtype: float32
  trust_remote_code: false
"""


def test_config_is_external_and_resolves_artifact_relative_to_yaml(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config" / "runtime.yaml"
    path.parent.mkdir()
    path.write_text(CONFIG, encoding="utf-8")

    config = AppConfig.from_yaml(path)

    assert config.dense.model_id == "org/embedding"
    assert config.reranker.model_id == "org/reranker"
    assert config.fusion.candidate_documents == 120
    assert config.sparse.query_encoder_manifest == (
        path.parent / "assets/sparse/manifest.json"
    ).resolve()


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(CONFIG + "unexpected: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown root"):
        AppConfig.from_yaml(path)


def test_optional_llm_config_uses_environment_reference_not_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = """
llm:
  provider: openai_compatible
  model: legal-generation-model
  base_url: https://llm.example.test/v1/
  api_key_env: LEGAL_RAG_LLM_API_KEY
  timeout_seconds: 60
  max_retries: 2
  temperature: 0.0
  max_output_tokens: 1024
  structured_output_mode: json_schema
"""
    path = tmp_path / "runtime.yaml"
    path.write_text(CONFIG + llm, encoding="utf-8")
    monkeypatch.setenv("LEGAL_RAG_LLM_API_KEY", "runtime-secret")

    config = AppConfig.from_yaml(path)

    assert config.llm is not None
    assert config.llm.model == "legal-generation-model"
    assert config.llm.base_url == "https://llm.example.test/v1"
    assert config.llm.api_key() == "runtime-secret"


def test_optional_rag_config_controls_context_and_evidence(tmp_path: Path) -> None:
    rag = """
rag:
  max_context_tokens: 4096
  max_chunks: 6
  max_chunks_per_document: null
  include_legal_metadata: true
  merge_adjacent_chunks: true
  evidence:
    min_sources: 2
    min_top_reranker_score: 1.5
    min_top_score_margin: 0.2
"""
    path = tmp_path / "runtime.yaml"
    path.write_text(CONFIG + rag, encoding="utf-8")

    config = AppConfig.from_yaml(path)

    assert config.rag.max_context_tokens == 4096
    assert config.rag.max_chunks == 6
    assert config.rag.max_chunks_per_document is None
    assert config.rag.merge_adjacent_chunks is True
    assert config.rag.evidence.min_sources == 2
    assert config.rag.evidence.min_top_reranker_score == 1.5
    assert config.rag.evidence.min_top_score_margin == 0.2
