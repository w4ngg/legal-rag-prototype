from __future__ import annotations

import hashlib
import json
from pathlib import Path

from legal_ir_prototype.core import SparseConfig
from legal_ir_prototype.model import SparseEncoder


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def test_sparse_query_uses_term_frequency_and_drops_oov(tmp_path: Path) -> None:
    vocabulary_path = tmp_path / "vocab.index.json"
    _write_json(vocabulary_path, {"điều": 0, "luật": 1, "khiếu": 2, "nại": 3})
    checksum = hashlib.sha256(vocabulary_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    _write_json(
        manifest_path,
        {
            "format_version": 1,
            "artifact_type": "bm25s_sparse_query_encoder",
            "collection_name": "test",
            "vector_name": "bm25_sparse",
            "scoring": {
                "document_weighting": "bm25s_lucene_precomputed",
                "query_weighting": "token_frequency",
                "qdrant_modifier": None,
                "similarity": "dot",
            },
            "tokenizer": {
                "library": "bm25s",
                "lower": True,
                "token_pattern": r"(?u)\b\w\w+\b",
                "stopwords": None,
                "stemmer": None,
                "drop_out_of_vocabulary_tokens": True,
            },
            "vocabulary": {
                "filename": vocabulary_path.name,
                "sha256": checksum,
                "size": 4,
                "indexed_dimension": 4,
            },
            "bm25": {"filename": "params.index.json", "parameters": {}},
        },
    )

    encoder = SparseEncoder(SparseConfig(manifest_path))
    vector = encoder.encode("Điều luật điều không-có-trong-vocab")

    assert encoder.vector_name == "bm25_sparse"
    assert vector.indices == (0, 1)
    assert vector.values == (2.0, 1.0)
