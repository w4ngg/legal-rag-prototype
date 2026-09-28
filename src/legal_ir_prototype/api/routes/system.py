from __future__ import annotations

"""Liveness, readiness, metrics and build identity routes."""

import asyncio

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST

from ..dependencies import get_metrics, get_request_id
from ..metrics import APIMetrics
from ..runtime import ReadinessReport
from ..schemas import HealthResponse, ReadyResponse, VersionResponse


router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={503: {"model": ReadyResponse}},
)
async def ready(request: Request) -> ReadyResponse | JSONResponse:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        report = ReadinessReport(
            embedding=False,
            reranker=False,
            qdrant=False,
            cuda=not request.app.state.require_cuda,
            llm=False,
        )
    else:
        report = await asyncio.to_thread(runtime.readiness)
    payload = ReadyResponse(**report.to_dict())
    if not report.ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload.model_dump(),
        )
    return payload


@router.get("/metrics", include_in_schema=False)
def metrics(metrics: APIMetrics = Depends(get_metrics)) -> Response:
    return Response(content=metrics.render(), media_type=CONTENT_TYPE_LATEST)


@router.get("/version", response_model=VersionResponse)
def version_info(
    request: Request,
    request_id: str = Depends(get_request_id),
) -> VersionResponse | JSONResponse:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "request_id": request_id,
                "error": {
                    "code": "VERSION_UNAVAILABLE",
                    "message": "Chưa tải được cấu hình phiên bản pipeline",
                    "retryable": True,
                },
            },
        )
    return VersionResponse(request_id=request_id, **runtime.version_info())
