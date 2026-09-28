from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping, Sequence, TypeVar

from ..core.config import QdrantConfig
from ..domain.models import LegalChunk, RetrievalHit, SparseQueryVector
from .errors import QdrantUnavailableError


_CORE_PAYLOAD_FIELDS = {
    "chunk_id",
    "document_id",
    "passage",
    "retrieval_text",
}
_T = TypeVar("_T")


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        result = value.model_dump()
    elif hasattr(value, "dict"):
        result = value.dict()
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise ValueError(f"cannot inspect Qdrant response type: {type(value)!r}")
    if not isinstance(result, dict):
        raise ValueError("unexpected Qdrant response")
    return result


class QdrantVectorStore:
    """Named dense/sparse vector search over the ingested long-chunk collection."""

    def __init__(self, config: QdrantConfig, *, client: Any | None = None) -> None:
        self.config = config
        self._client = client
        self._schema_validated = False

    def _get_client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key(),
                prefer_grpc=self.config.prefer_grpc,
                grpc_port=self.config.grpc_port,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    @staticmethod
    def _call(operation: str, callback: Callable[[], _T]) -> _T:
        """Normalize Qdrant transport/service failures at the adapter boundary.

        Schema and payload validation deliberately happen outside this wrapper so
        programming/data-contract errors still surface as internal errors instead
        of being mislabeled as transient dependency failures.
        """

        try:
            return callback()
        except QdrantUnavailableError:
            raise
        except Exception as exc:
            raise QdrantUnavailableError(operation) from exc

    def validate_schema(self, *, expected_dense_dimension: int) -> None:
        if self._schema_validated:
            return
        collection = self._call(
            "get_collection",
            lambda: self._get_client().get_collection(self.config.collection),
        )
        info = _model_dump(collection)
        params = info.get("config", {}).get("params", {})
        vectors = params.get("vectors", {})
        dense = vectors.get(self.config.dense_vector_name)
        if not isinstance(dense, Mapping):
            raise ValueError(
                f"Qdrant collection has no dense vector "
                f"{self.config.dense_vector_name!r}"
            )
        if int(dense.get("size", -1)) != expected_dense_dimension:
            raise ValueError(
                f"Qdrant dense dimension is {dense.get('size')}; expected "
                f"{expected_dense_dimension}"
            )
        distance = str(dense.get("distance", "")).casefold()
        if distance not in {"dot", "distance.dot"}:
            raise ValueError(
                f"Qdrant dense distance is {dense.get('distance')!r}; expected DOT"
            )
        sparse_vectors = params.get("sparse_vectors") or {}
        if self.config.sparse_vector_name not in sparse_vectors:
            raise ValueError(
                f"Qdrant collection has no sparse vector "
                f"{self.config.sparse_vector_name!r}"
            )
        self._schema_validated = True

    @property
    def is_schema_validated(self) -> bool:
        return self._schema_validated

    def check_connection(self) -> None:
        """Raise when the configured collection cannot be reached."""

        self._call(
            "get_collection",
            lambda: self._get_client().get_collection(self.config.collection),
        )

    @staticmethod
    def _chunk_from_point(point: Any) -> LegalChunk:
        payload = getattr(point, "payload", None)
        if not isinstance(payload, Mapping):
            raise ValueError("Qdrant result is missing its chunk payload")
        required = {
            "schema_version",
            "chunk_id",
            "document_id",
            "vector_row",
            "granularity",
            "passage",
            "retrieval_text",
            "passage_sha256",
            "embedding_text_sha256",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"Qdrant chunk payload is missing fields: {missing}")
        if int(payload["schema_version"]) != 1:
            raise ValueError(
                f"unsupported Qdrant payload schema: {payload['schema_version']}"
            )
        if int(payload["vector_row"]) < 0:
            raise ValueError("Qdrant payload vector_row must be non-negative")
        if str(payload["granularity"]) != "long":
            raise ValueError("prototype collection must contain long chunks")
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in _CORE_PAYLOAD_FIELDS
        }
        point_id = getattr(point, "id", None)
        if point_id is not None:
            metadata["qdrant_point_id"] = str(point_id)
        return LegalChunk(
            chunk_id=str(payload.get("chunk_id", "")),
            document_id=str(payload.get("document_id", "")),
            passage=str(payload.get("passage", "")),
            retrieval_text=str(payload.get("retrieval_text", "")),
            metadata=metadata,
        )

    def _query(
        self,
        *,
        query: Any,
        using: str,
        top_k: int,
        channel: str,
    ) -> list[RetrievalHit]:
        if top_k <= 0:
            return []
        from qdrant_client import models

        response = self._call(
            f"query_points:{channel}",
            lambda: self._get_client().query_points(
                collection_name=self.config.collection,
                query=query,
                using=using,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                search_params=models.SearchParams(
                    exact=self.config.exact,
                    hnsw_ef=self.config.hnsw_ef,
                ),
            ),
        )
        points: Sequence[Any] = getattr(response, "points", response)
        hits: list[RetrievalHit] = []
        for point in points:
            score = float(getattr(point, "score"))
            # Research BM25 removes padding/non-matching rows. Dense inner
            # products are retained regardless of sign.
            if channel == "sparse" and score <= 0.0:
                continue
            hits.append(
                RetrievalHit(
                    chunk=self._chunk_from_point(point),
                    channel=channel,
                    score=score,
                )
            )
        return hits

    def search_dense(self, vector: Sequence[float], top_k: int) -> list[RetrievalHit]:
        return self._query(
            query=list(vector),
            using=self.config.dense_vector_name,
            top_k=top_k,
            channel="dense",
        )

    def search_sparse(
        self,
        vector: SparseQueryVector,
        top_k: int,
    ) -> list[RetrievalHit]:
        if not vector.indices:
            return []
        from qdrant_client import models

        return self._query(
            query=models.SparseVector(
                indices=list(vector.indices),
                values=list(vector.values),
            ),
            using=self.config.sparse_vector_name,
            top_k=top_k,
            channel="sparse",
        )
