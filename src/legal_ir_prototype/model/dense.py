from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.config import DenseConfig
from ..core.runtime import detected_torch_device, inference_torch_dtype, runtime_device


class DenseEncoder:
    """Pinned Vietnamese_Embedding_v2 query encoder used by the research path."""

    def __init__(self, config: DenseConfig) -> None:
        self.config = config
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        import torch
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self.config.model_id,
            revision=self.config.revision,
            device=runtime_device(self.config.device),
            trust_remote_code=self.config.trust_remote_code,
            model_kwargs={
                "torch_dtype": inference_torch_dtype(
                    torch, self.config.dtype, self.config.device
                )
            },
        )
        self._model.max_seq_length = self.config.max_length
        dimension = int(self._model.get_sentence_embedding_dimension())
        if dimension != self.config.dimension:
            raise ValueError(
                f"dense model dimension is {dimension}; expected "
                f"{self.config.dimension}"
            )
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load and validate the pinned model without running a query."""

        self._load()

    def encode(self, query: str) -> np.ndarray:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        import torch

        vector = self._load().encode_query(
            [normalized_query],
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            device=detected_torch_device(torch, self.config.device),
        )
        result = np.asarray(vector, dtype=np.float32)
        if result.shape != (1, self.config.dimension):
            raise ValueError(f"unexpected dense query shape: {result.shape}")
        if not np.isfinite(result).all():
            raise ValueError("dense query vector contains NaN or infinite values")
        return np.ascontiguousarray(result[0], dtype=np.float32)

    def encode_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode corpus passages with the model's document prompt/path.

        Query and document encoding are intentionally separate because the
        pinned Vietnamese embedding model exposes asymmetric
        ``encode_query``/``encode_document`` entry points.
        """

        normalized = [str(text).strip() for text in texts]
        if any(not text for text in normalized):
            raise ValueError("document texts must not contain empty values")
        if not normalized:
            return np.empty((0, self.config.dimension), dtype=np.float32)
        import torch

        vectors = self._load().encode_document(
            normalized,
            batch_size=batch_size or self.config.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            device=detected_torch_device(torch, self.config.device),
        )
        result = np.asarray(vectors, dtype=np.float32)
        expected = (len(normalized), self.config.dimension)
        if result.shape != expected:
            raise ValueError(
                f"unexpected dense document shape: {result.shape}; expected {expected}"
            )
        if not np.isfinite(result).all():
            raise ValueError("dense document vectors contain NaN or infinite values")
        return np.ascontiguousarray(result, dtype=np.float32)
