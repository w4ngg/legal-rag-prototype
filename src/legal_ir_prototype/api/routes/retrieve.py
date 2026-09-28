from __future__ import annotations

"""Authenticated internal retrieval diagnostics route."""

import asyncio
import time

from fastapi import APIRouter, Depends, Request

from ..dependencies import (
    get_metrics,
    get_request_id,
    get_runtime,
    require_internal_access,
)
from ..errors import APIError
from ..metrics import APIMetrics
from ..runtime import ApplicationRuntime
from ..schemas import RetrieveRequest, RetrieveResponse, RetrievalResultResponse


router = APIRouter(prefix="/v1", tags=["internal"])


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    dependencies=[Depends(require_internal_access)],
)
async def retrieve(
    payload: RetrieveRequest,
    request: Request,
    runtime: ApplicationRuntime = Depends(get_runtime),
    metrics: APIMetrics = Depends(get_metrics),
    request_id: str = Depends(get_request_id),
) -> RetrieveResponse:
    configured_top_k = runtime.config.reranker.output_top_k
    if payload.top_k > configured_top_k:
        raise APIError(
            status_code=422,
            code="TOP_K_EXCEEDS_PIPELINE_OUTPUT",
            message=(
                f"top_k không được vượt reranker.output_top_k="
                f"{configured_top_k}"
            ),
            retryable=False,
        )

    started = time.perf_counter()
    try:
        async with request.app.state.inference_semaphore:
            chunks = await asyncio.to_thread(
                runtime.retrieval_service.retrieve,
                payload.query,
            )
    except Exception:
        metrics.retrieval_requests.labels(outcome="error").inc()
        raise
    metrics.retrieval_duration.observe(time.perf_counter() - started)
    metrics.retrieval_requests.labels(outcome="success").inc()

    results = [
        RetrievalResultResponse(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            text=chunk.passage,
            metadata=dict(chunk.metadata),
            retrieval_scores=dict(chunk.retrieval_scores),
            retrieval_ranks=dict(chunk.retrieval_ranks),
            fusion_score=chunk.fusion_score,
            fusion_rank=chunk.fusion_rank,
            reranker_score=chunk.reranker_score,
            reranker_rank=chunk.reranker_rank,
        )
        for chunk in chunks[: payload.top_k]
    ]
    return RetrieveResponse(
        request_id=request_id,
        result_count=len(results),
        results=results,
    )
