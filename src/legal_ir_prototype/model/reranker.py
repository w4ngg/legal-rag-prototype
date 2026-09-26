from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.config import RerankerConfig
from ..core.runtime import inference_torch_dtype, runtime_device
from ..domain.models import LegalChunk


class Reranker:
    """Fine-tuned Vietnamese cross-encoder returning uncalibrated raw logits."""

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        import torch
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            self.config.model_id,
            revision=self.config.revision,
            device=runtime_device(self.config.device),
            trust_remote_code=self.config.trust_remote_code,
            model_kwargs={
                "torch_dtype": inference_torch_dtype(
                    torch, self.config.dtype, self.config.device
                )
            },
            max_length=self.config.max_length,
            # The research implementation ranks raw one-label logits. Do not
            # apply sigmoid or interpret the score as calibrated confidence.
            activation_fn=torch.nn.Identity(),
        )
        return self._model

    def score(self, query: str, chunks: Sequence[LegalChunk]) -> list[float]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if not chunks:
            return []
        pairs = [(normalized_query, chunk.model_text) for chunk in chunks]
        values = self._load().predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = [float(value) for value in np.asarray(values).reshape(-1)]
        if len(scores) != len(chunks):
            raise ValueError(
                f"reranker returned {len(scores)} scores for {len(chunks)} chunks"
            )
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("reranker returned a non-finite score")
        return scores
