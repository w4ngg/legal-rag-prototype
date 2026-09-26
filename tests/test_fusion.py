from __future__ import annotations

import pytest

from legal_ir_prototype.core import FusionConfig
from legal_ir_prototype.domain import LegalChunk, RetrievalHit
from legal_ir_prototype.fusion import FusionStrategy


def _chunk(chunk_id: str, document_id: str | None = None) -> LegalChunk:
    return LegalChunk(
        chunk_id=chunk_id,
        document_id=document_id or chunk_id.split(":", 1)[0],
        passage=f"passage {chunk_id}",
        retrieval_text=f"text {chunk_id}",
    )


def _hit(channel: str, chunk_id: str, score: float) -> RetrievalHit:
    return RetrievalHit(chunk=_chunk(chunk_id), channel=channel, score=score)


def test_weighted_rrf_deduplicates_and_preserves_raw_lane_scores() -> None:
    strategy = FusionStrategy(
        FusionConfig(
            "weighted_rrf",
            60,
            dense_weight=1.0,
            sparse_weight=1.0,
            candidate_documents=None,
            evidence_chunks_per_document=None,
        )
    )
    results = strategy.fuse(
        {
            "sparse": [
                _hit("sparse", "a:0", 9.0),
                _hit("sparse", "a:0", 8.0),
                _hit("sparse", "b:0", 7.0),
            ],
            "dense": [
                _hit("dense", "b:0", 0.8),
                _hit("dense", "c:0", 0.7),
            ],
        },
    )

    assert [item.chunk_id for item in results] == ["b:0", "a:0", "c:0"]
    assert results[0].retrieval_ranks == {"sparse": 2, "dense": 1}
    assert results[0].retrieval_scores == {"sparse": 7.0, "dense": 0.8}
    assert results[0].fusion_score == pytest.approx(1 / 62 + 1 / 61)
    assert len(results) == 3


@pytest.mark.parametrize(
    ("sparse", "dense", "expected"),
    [
        (
            [
                ("194863:x", 19.2883),
                ("68961:x", 19.2465),
                ("160120:x187", 19.2383),
            ],
            [
                ("160120:x187", 0.6540),
                ("102434:x", 0.6455),
                ("160120:x188", 0.6426),
            ],
            ["160120:x187", "160120:x188", "194863:x", "102434:x", "68961:x"],
        ),
        (
            [
                ("209403:x", 22.4191),
                ("152976:x", 20.3054),
                ("43535:x", 19.9862),
            ],
            [
                ("199066:x", 0.5831),
                ("58662:x", 0.5022),
                ("8747:x", 0.5017),
            ],
            ["199066:x", "209403:x", "152976:x", "58662:x", "43535:x", "8747:x"],
        ),
        (
            [
                ("157284:x006", 28.0227),
                ("281469:x020", 27.0046),
                ("281469:x021", 26.8341),
            ],
            [
                ("157284:x006", 0.6097),
                ("270884:x011", 0.5058),
                ("157284:x024", 0.4961),
            ],
            [
                "157284:x006",
                "157284:x024",
                "270884:x011",
                "281469:x020",
                "281469:x021",
            ],
        ),
    ],
)
def test_representative_research_lane_outputs_have_stable_rrf_order(
    sparse: list[tuple[str, float]],
    dense: list[tuple[str, float]],
    expected: list[str],
) -> None:
    """Regression cases derived from long-only research diagnostics."""

    strategy = FusionStrategy(
        FusionConfig("weighted_rrf", 60, 1.0, 1.0, None, None)
    )
    results = strategy.fuse(
        {
            "sparse": [_hit("sparse", chunk_id, score) for chunk_id, score in sparse],
            "dense": [_hit("dense", chunk_id, score) for chunk_id, score in dense],
        },
    )
    assert [item.chunk_id for item in results] == expected
