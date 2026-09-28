from __future__ import annotations

"""FastAPI dependency accessors and internal endpoint guard."""

import hmac

from fastapi import Header, Request

from .errors import APIError
from .metrics import APIMetrics
from .runtime import ApplicationRuntime


def get_request_id(request: Request) -> str:
    return str(request.state.request_id)


def get_metrics(request: Request) -> APIMetrics:
    return request.app.state.metrics


def get_runtime(request: Request) -> ApplicationRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    startup_error = getattr(request.app.state, "startup_error", None)
    if runtime is None or startup_error is not None:
        raise APIError(
            status_code=503,
            code="SERVICE_NOT_READY",
            message="Dịch vụ chưa sẵn sàng nhận yêu cầu",
            retryable=True,
        )
    return runtime


def require_internal_access(
    request: Request,
    x_internal_api_key: str | None = Header(default=None),
) -> None:
    if not request.app.state.enable_retrieve_endpoint:
        raise APIError(
            status_code=404,
            code="NOT_FOUND",
            message="Endpoint không tồn tại",
            retryable=False,
        )
    expected = request.app.state.internal_api_key
    if expected is None:
        raise APIError(
            status_code=503,
            code="INTERNAL_ENDPOINT_NOT_CONFIGURED",
            message="Endpoint nội bộ chưa được cấu hình khóa truy cập",
            retryable=False,
        )
    if x_internal_api_key is None or not hmac.compare_digest(
        x_internal_api_key,
        expected,
    ):
        raise APIError(
            status_code=401,
            code="UNAUTHORIZED",
            message="Không có quyền truy cập endpoint nội bộ",
            retryable=False,
        )
