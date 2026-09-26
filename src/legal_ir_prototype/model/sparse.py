from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import bm25s

from ..core.config import SparseConfig
from ..domain.models import SparseQueryVector


def _load_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SparseEncoder:
    """Exact BM25S vocabulary/token-frequency query encoder for Qdrant."""

    def __init__(self, config: SparseConfig) -> None:
        self.config = config
        self.manifest = _load_object(config.query_encoder_manifest)
        self._validate_manifest(self.manifest)
        vocab_info = self.manifest["vocabulary"]
        vocabulary_path = config.query_encoder_manifest.parent / str(
            vocab_info["filename"]
        )
        if _sha256(vocabulary_path) != str(vocab_info["sha256"]):
            raise ValueError("sparse vocabulary checksum does not match manifest")
        vocabulary = _load_object(vocabulary_path)
        self._vocabulary = {str(token): int(index) for token, index in vocabulary.items()}
        self._indexed_dimension = int(vocab_info["indexed_dimension"])
        if len(self._vocabulary) != int(vocab_info["size"]):
            raise ValueError("sparse vocabulary size does not match manifest")

    @property
    def vector_name(self) -> str:
        return str(self.manifest["vector_name"])

    @staticmethod
    def _validate_manifest(manifest: Mapping[str, Any]) -> None:
        if manifest.get("artifact_type") != "bm25s_sparse_query_encoder":
            raise ValueError("unsupported sparse query encoder artifact")
        scoring = manifest.get("scoring") or {}
        expected_scoring = {
            "document_weighting": "bm25s_lucene_precomputed",
            "query_weighting": "token_frequency",
            "qdrant_modifier": None,
            "similarity": "dot",
        }
        if any(scoring.get(key) != value for key, value in expected_scoring.items()):
            raise ValueError("sparse scoring manifest does not match BM25S/Qdrant path")
        tokenizer = manifest.get("tokenizer") or {}
        expected_tokenizer = {
            "library": "bm25s",
            "lower": True,
            "token_pattern": r"(?u)\b\w\w+\b",
            "stopwords": None,
            "stemmer": None,
            "drop_out_of_vocabulary_tokens": True,
        }
        if any(tokenizer.get(key) != value for key, value in expected_tokenizer.items()):
            raise ValueError("sparse tokenizer manifest differs from research BM25S")
        vocabulary = manifest.get("vocabulary") or {}
        for field in ("filename", "sha256", "size", "indexed_dimension"):
            if field not in vocabulary:
                raise ValueError(f"sparse vocabulary manifest is missing {field}")

    def encode(self, query: str) -> SparseQueryVector:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        tokens = bm25s.tokenize(
            [normalized_query],
            stopwords=None,
            stemmer=None,
            return_ids=False,
            show_progress=False,
        )
        # bm25s returns one token list per input. Document weights stored in
        # Qdrant are already BM25 Lucene weights, so the online query side is
        # deliberately only term frequency (no second IDF transformation).
        query_tokens = tokens[0]
        frequencies = Counter(
            index
            for token in query_tokens
            if (index := self._vocabulary.get(str(token))) is not None
            and 0 <= index < self._indexed_dimension
        )
        indices = tuple(sorted(frequencies))
        return SparseQueryVector(
            indices=indices,
            values=tuple(float(frequencies[index]) for index in indices),
        )
