from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import IndexConfig
from .schema import canonical_chunk_hash, iter_chunk_records, normalize_chunk_record


FORMAT_VERSION = 1


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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


@dataclass(frozen=True, slots=True)
class SparseCorpus:
    directory: Path
    manifest: dict[str, Any]
    chunk_count: int
    chunk_records_sha256: str
    vocabulary_size: int
    indexed_dimension: int
    nonzero_count: int

    @property
    def vocab_path(self) -> Path:
        return self.directory / "vocab.index.json"

    @property
    def params_path(self) -> Path:
        return self.directory / "params.index.json"

    def load_csr(self) -> Any:
        try:
            import scipy.sparse as sp
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "direct ingestion requires scipy; install with "
                "`python -m pip install -e '.[ingestion]'`"
            ) from exc
        data = np.load(
            self.directory / "data.csc.index.npy", mmap_mode="r", allow_pickle=False
        )
        indices = np.load(
            self.directory / "indices.csc.index.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        indptr = np.load(
            self.directory / "indptr.csc.index.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        matrix = sp.csc_matrix(
            (data, indices, indptr),
            shape=(self.chunk_count, self.indexed_dimension),
            copy=False,
        ).tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        return matrix


def _validate(directory: Path, expected_hash: str, expected_count: int) -> SparseCorpus:
    required = [
        directory / "manifest.json",
        directory / "params.index.json",
        directory / "vocab.index.json",
        directory / "data.csc.index.npy",
        directory / "indices.csc.index.npy",
        directory / "indptr.csc.index.npy",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "incomplete BM25 build state: " + ", ".join(str(path) for path in missing)
        )
    manifest = _load_json(directory / "manifest.json")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported BM25 build-state format")
    if manifest.get("chunk_records_sha256") != expected_hash:
        raise ValueError("BM25 build state belongs to different chunk content/order")
    if int(manifest.get("chunk_count", -1)) != expected_count:
        raise ValueError("BM25 build state has a different chunk count")
    for filename, checksum in (manifest.get("artifacts") or {}).items():
        path = directory / str(filename)
        if not path.is_file() or _sha256(path) != str(checksum):
            raise ValueError(f"BM25 artifact checksum mismatch: {path}")
    vocab = _load_json(directory / "vocab.index.json")
    params = _load_json(directory / "params.index.json")
    data = np.load(directory / "data.csc.index.npy", mmap_mode="r")
    indices = np.load(directory / "indices.csc.index.npy", mmap_mode="r")
    indptr = np.load(directory / "indptr.csc.index.npy", mmap_mode="r")
    if int(params.get("num_docs", -1)) != expected_count:
        raise ValueError("BM25 params contain a different document count")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("BM25 CSC artifacts must be one-dimensional")
    if len(data) != len(indices) or int(indptr[-1]) != len(data):
        raise ValueError("BM25 CSC artifacts are inconsistent")
    indexed_dimension = len(indptr) - 1
    if len(vocab) not in {indexed_dimension, indexed_dimension + 1}:
        raise ValueError("BM25 vocabulary and indexed dimension are incompatible")
    return SparseCorpus(
        directory=directory,
        manifest=manifest,
        chunk_count=expected_count,
        chunk_records_sha256=expected_hash,
        vocabulary_size=len(vocab),
        indexed_dimension=indexed_dimension,
        nonzero_count=len(data),
    )


def prepare_sparse_corpus(
    chunks_path: str | Path,
    state_dir: str | Path,
    config: IndexConfig,
    *,
    overwrite: bool = False,
) -> SparseCorpus:
    """Build resumable BM25 document weights used by Qdrant sparse vectors.

    The directory is ingestion state, not a second online search service.  It
    exists because BM25 document weights depend on whole-corpus statistics and
    because a restart must not repeat that global computation.
    """

    source = Path(chunks_path).expanduser().resolve()
    destination = Path(state_dir).expanduser().resolve()
    chunk_hash, chunk_count = canonical_chunk_hash(source)
    if chunk_count <= 0:
        raise ValueError("cannot build BM25 weights for an empty chunk corpus")
    if (destination / "manifest.json").is_file() and not overwrite:
        corpus = _validate(destination, chunk_hash, chunk_count)
        expected = {
            "method": config.bm25_method,
            "k1": config.bm25_k1,
            "b": config.bm25_b,
        }
        if corpus.manifest.get("bm25") != expected:
            raise ValueError(
                "existing BM25 build state uses different parameters; "
                "use a new state directory or --overwrite"
            )
        return corpus

    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"BM25 build-state directory is not empty: {destination}"
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    import bm25s

    texts = [
        normalize_chunk_record(record)["retrieval_text"]
        for record in iter_chunk_records(source)
    ]
    tokens = bm25s.tokenize(
        texts,
        stopwords=None,
        stemmer=None,
        show_progress=True,
    )
    model = bm25s.BM25(
        method=config.bm25_method,
        k1=config.bm25_k1,
        b=config.bm25_b,
    )
    model.index(tokens, show_progress=True)
    model.save(destination, show_progress=True)
    artifact_names = [
        "params.index.json",
        "vocab.index.json",
        "data.csc.index.npy",
        "indices.csc.index.npy",
        "indptr.csc.index.npy",
    ]
    _write_json(
        destination / "manifest.json",
        {
            "format_version": FORMAT_VERSION,
            "artifact_type": "bm25s_qdrant_ingestion_state",
            "chunk_count": chunk_count,
            "chunk_records_sha256": chunk_hash,
            "bm25": {
                "method": config.bm25_method,
                "k1": config.bm25_k1,
                "b": config.bm25_b,
            },
            "artifacts": {
                name: _sha256(destination / name) for name in artifact_names
            },
        },
    )
    return _validate(destination, chunk_hash, chunk_count)


def export_query_bundle(
    corpus: SparseCorpus,
    output_dir: str | Path,
    *,
    collection_name: str,
    vector_name: str,
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for source in (corpus.vocab_path, corpus.params_path):
        target = destination / source.name
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    manifest_path = destination / "manifest.json"
    _write_json(
        manifest_path,
        {
            "format_version": 1,
            "artifact_type": "bm25s_sparse_query_encoder",
            "collection_name": collection_name,
            "vector_name": vector_name,
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
                "filename": "vocab.index.json",
                "sha256": _sha256(destination / "vocab.index.json"),
                "size": corpus.vocabulary_size,
                "indexed_dimension": corpus.indexed_dimension,
            },
            "bm25": {
                "filename": "params.index.json",
                "sha256": _sha256(destination / "params.index.json"),
                "parameters": _load_json(destination / "params.index.json"),
            },
        },
    )
    return manifest_path
