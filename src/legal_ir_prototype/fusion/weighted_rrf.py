from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

from ..core.config import FusionConfig
from ..domain.models import LegalChunk, RetrievalHit


class FusionStrategy:
    """Research-compatible document RRF and evidence-chunk selection."""

    def __init__(self, config: FusionConfig) -> None:
        self.config = config
        self._weights = {
            "dense": config.dense_weight,
            "sparse": config.sparse_weight,
        }

    @staticmethod
    def _same_chunk(left: LegalChunk, right: LegalChunk) -> bool:
        return (
            left.chunk_id == right.chunk_id
            and left.document_id == right.document_id
            and left.passage == right.passage
            and left.retrieval_text == right.retrieval_text
        )

    @staticmethod
    def _rank_channel(
        channel: str,
        hits: Sequence[RetrievalHit],
    ) -> list[RetrievalHit]:
        """Match research `rank_chunks`: best duplicate, score, then chunk ID."""

        best: dict[str, RetrievalHit] = {}
        for hit in hits:
            if hit.channel != channel:
                raise ValueError(
                    f"hit channel {hit.channel!r} does not match {channel!r}"
                )
            if not math.isfinite(hit.score):
                continue
            previous = best.get(hit.chunk.chunk_id)
            if previous is not None and not FusionStrategy._same_chunk(
                previous.chunk, hit.chunk
            ):
                raise ValueError(
                    f"conflicting payloads for chunk {hit.chunk.chunk_id}"
                )
            if previous is None or hit.score > previous.score:
                best[hit.chunk.chunk_id] = hit
        return sorted(best.values(), key=lambda hit: (-hit.score, hit.chunk.chunk_id))

    @staticmethod
    def _rank_documents(ranked_hits: Sequence[RetrievalHit]) -> dict[str, int]:
        """Match research lane-local chunk-to-document MaxP ranking."""

        best: dict[str, RetrievalHit] = {}
        for hit in ranked_hits:
            previous = best.get(hit.chunk.document_id)
            if previous is None or hit.score > previous.score:
                best[hit.chunk.document_id] = hit
        ordered = sorted(
            best.items(),
            key=lambda item: (-item[1].score, item[0]),
        )
        return {
            document_id: rank
            for rank, (document_id, _) in enumerate(ordered, start=1)
        }

    def fuse(
        self,
        channel_hits: Mapping[str, Sequence[RetrievalHit]],
    ) -> list[LegalChunk]:
        unsupported = sorted(set(channel_hits) - set(self._weights))
        if unsupported:
            raise ValueError(f"unsupported fusion channels: {unsupported}")

        ranked_channels: dict[str, list[RetrievalHit]] = {}
        chunks: dict[str, LegalChunk] = {}
        raw_scores: dict[str, dict[str, float]] = defaultdict(dict)
        chunk_ranks: dict[str, dict[str, int]] = defaultdict(dict)
        document_ranks: dict[str, dict[str, int]] = defaultdict(dict)

        for channel in ("sparse", "dense"):
            ranked = self._rank_channel(channel, channel_hits.get(channel, ()))
            ranked_channels[channel] = ranked
            for rank, hit in enumerate(ranked, start=1):
                chunk_id = hit.chunk.chunk_id
                existing = chunks.get(chunk_id)
                if existing is not None and not self._same_chunk(existing, hit.chunk):
                    raise ValueError(f"conflicting payloads for chunk {chunk_id}")
                chunks[chunk_id] = hit.chunk
                raw_scores[chunk_id][channel] = hit.score
                chunk_ranks[chunk_id][channel] = rank
            for document_id, rank in self._rank_documents(ranked).items():
                document_ranks[document_id][channel] = rank

        # First RRF level: document candidates after lane-local MaxP.
        document_scores: dict[str, float] = {}
        for document_id, ranks in document_ranks.items():
            document_scores[document_id] = sum(
                self._weights[channel] / (self.config.rrf_k + rank)
                for channel, rank in ranks.items()
                if self._weights[channel] > 0
            )
        ordered_documents = sorted(
            document_scores,
            key=lambda document_id: (-document_scores[document_id], document_id),
        )
        if self.config.candidate_documents is not None:
            ordered_documents = ordered_documents[: self.config.candidate_documents]
        retained_documents = set(ordered_documents)
        document_fusion_ranks = {
            document_id: rank
            for rank, document_id in enumerate(ordered_documents, start=1)
        }

        # Second RRF level: evidence chunks inside each retained document.
        evidence_scores: dict[str, float] = defaultdict(float)
        chunks_by_document: dict[str, set[str]] = defaultdict(set)
        for channel, ranked in ranked_channels.items():
            weight = self._weights[channel]
            if weight <= 0:
                continue
            for rank, hit in enumerate(ranked, start=1):
                if hit.chunk.document_id not in retained_documents:
                    continue
                evidence_scores[hit.chunk.chunk_id] += weight / (
                    self.config.rrf_k + rank
                )
                chunks_by_document[hit.chunk.document_id].add(hit.chunk.chunk_id)

        selected: list[tuple[str, int]] = []
        for document_id in ordered_documents:
            evidence_ids = sorted(
                chunks_by_document[document_id],
                key=lambda chunk_id: (-evidence_scores[chunk_id], chunk_id),
            )
            if self.config.evidence_chunks_per_document is not None:
                evidence_ids = evidence_ids[
                    : self.config.evidence_chunks_per_document
                ]
            selected.extend(
                (chunk_id, evidence_rank)
                for evidence_rank, chunk_id in enumerate(evidence_ids, start=1)
            )

        results: list[LegalChunk] = []
        for selection_rank, (chunk_id, evidence_rank) in enumerate(
            selected, start=1
        ):
            chunk = chunks[chunk_id]
            document_id = chunk.document_id
            metadata = dict(chunk.metadata)
            metadata["research_fusion"] = {
                "document_score": document_scores[document_id],
                "document_rank": document_fusion_ranks[document_id],
                "document_channel_ranks": dict(document_ranks[document_id]),
                "evidence_rank_within_document": evidence_rank,
                "candidate_sequence_rank": selection_rank,
            }
            results.append(
                chunk.with_scores(
                    metadata=metadata,
                    retrieval_scores=raw_scores[chunk_id],
                    retrieval_ranks=chunk_ranks[chunk_id],
                    fusion_score=evidence_scores[chunk_id],
                    fusion_rank=selection_rank,
                )
            )
        return results
