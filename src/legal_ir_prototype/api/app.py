from __future__ import annotations

"""FastAPI application factory and production server entry point."""

import argparse
import asyncio
import copy
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..core import AppConfig
from ..environment import load_environment_file
from ..vectorstore import QdrantUnavailableError
from .errors import APIError
from .metrics import APIMetrics
from .observability import (
    begin_request_timings,
    end_request_timings,
    request_timings_snapshot,
)
from .routes import chat, retrieve, system
from .runtime import ApplicationRuntime


LOGGER = logging.getLogger(__name__)
REQUEST_LOGGER = logging.getLogger("legal_ir_prototype.api.requests")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _environment_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _normalize_cors_origins(origins: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_origin in origins:
        origin = raw_origin.strip().rstrip("/")
        if not origin:
            continue
        if origin == "*":
            raise ValueError("LEGAL_RAG_CORS_ORIGINS must not contain '*'")
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "LEGAL_RAG_CORS_ORIGINS contains an invalid origin: "
                f"{raw_origin!r}"
            )
        if origin not in normalized:
            normalized.append(origin)
    return tuple(normalized)


def _environment_cors_origins() -> tuple[str, ...]:
    raw = os.environ.get("LEGAL_RAG_CORS_ORIGINS", "")
    return _normalize_cors_origins(raw.split(","))


def _error_body(
    request: Request,
    *,
    code: str,
    message: str,
    retryable: bool,
) -> dict[str, object]:
    return {
        "request_id": getattr(request.state, "request_id", "unavailable"),
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    }


def create_app(
    *,
    config_path: str | Path | None = None,
    runtime: ApplicationRuntime | None = None,
    enable_retrieve_endpoint: bool | None = None,
    internal_api_key: str | None = None,
    require_cuda: bool | None = None,
    max_concurrent_inference: int | None = None,
    cors_origins: Sequence[str] | None = None,
) -> FastAPI:
    enable_retrieve = (
        _environment_bool("LEGAL_RAG_ENABLE_RETRIEVE", False)
        if enable_retrieve_endpoint is None
        else bool(enable_retrieve_endpoint)
    )
    key = (
        os.environ.get("LEGAL_RAG_INTERNAL_API_KEY")
        if internal_api_key is None
        else internal_api_key
    )
    key = key.strip() if key and key.strip() else None
    if enable_retrieve and key is None:
        raise ValueError(
            "LEGAL_RAG_ENABLE_RETRIEVE=true requires "
            "LEGAL_RAG_INTERNAL_API_KEY"
        )
    cuda_required = (
        _environment_bool("LEGAL_RAG_REQUIRE_CUDA", False)
        if require_cuda is None
        else bool(require_cuda)
    )
    concurrency = max_concurrent_inference
    if concurrency is None:
        concurrency = int(
            os.environ.get("LEGAL_RAG_MAX_CONCURRENT_INFERENCE", "1")
        )
    if concurrency <= 0:
        raise ValueError("max_concurrent_inference must be positive")
    selected_config = Path(
        config_path
        or os.environ.get("LEGAL_RAG_CONFIG", "configs/runtime.yaml")
    ).expanduser()
    allowed_origins = (
        _environment_cors_origins()
        if cors_origins is None
        else _normalize_cors_origins(cors_origins)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        selected_runtime = runtime
        try:
            if selected_runtime is None:
                config = AppConfig.from_yaml(selected_config)
                selected_runtime = ApplicationRuntime.build(
                    config,
                    require_cuda=cuda_required,
                )
            app.state.runtime = selected_runtime
            bind_metrics = getattr(selected_runtime, "bind_metrics", None)
            if bind_metrics is not None:
                bind_metrics(app.state.metrics)
            await asyncio.to_thread(selected_runtime.startup)
            app.state.startup_error = None
        except Exception as exc:
            app.state.startup_error = exc
            LOGGER.exception("API dependencies failed to start")
        try:
            yield
        finally:
            active_runtime = getattr(app.state, "runtime", None)
            if active_runtime is not None:
                await active_runtime.aclose()

    app = FastAPI(
        title="Vietnamese Legal RAG API",
        version="1.0.0",
        lifespan=lifespan,
    )
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID"],
        )
    app.state.runtime = None
    app.state.startup_error = None
    app.state.metrics = APIMetrics()
    app.state.enable_retrieve_endpoint = enable_retrieve
    app.state.internal_api_key = key
    app.state.require_cuda = cuda_required
    app.state.inference_semaphore = asyncio.Semaphore(concurrency)

    @app.middleware("http")
    async def request_observability(request: Request, call_next):
        log_completion = (
            request.method == "POST" and request.url.path == "/v1/chat"
        )
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = (
            supplied
            if _REQUEST_ID.fullmatch(supplied)
            else f"req_{uuid.uuid4().hex}"
        )
        request.state.request_outcome = None
        request.state.error_code = None
        started = time.perf_counter()
        status_code = 500
        token = begin_request_timings()
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request.state.request_id
            return response
        finally:
            if log_completion:
                total_ms = (time.perf_counter() - started) * 1_000.0
                timings = request_timings_snapshot()
                outcome = request.state.request_outcome
                if outcome is None:
                    outcome = "success" if status_code < 400 else "error"

                def milliseconds(stage: str) -> float | None:
                    duration = timings.get(stage)
                    return (
                        None
                        if duration is None
                        else round(duration * 1_000.0, 3)
                    )

                event = {
                    "event": "http_request_completed",
                    "request_id": request.state.request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "retrieval_ms": milliseconds("retrieval_pre_rerank"),
                    "retrieval_total_ms": milliseconds("retrieval"),
                    "reranker_ms": milliseconds("reranker"),
                    "context_builder_ms": milliseconds("context_builder"),
                    "llm_ms": milliseconds("llm"),
                    "validation_ms": milliseconds("validation"),
                    "qdrant_dense_ms": milliseconds("qdrant_dense"),
                    "qdrant_sparse_ms": milliseconds("qdrant_sparse"),
                    "total_ms": round(total_ms, 3),
                    "status": outcome,
                    "http_status": status_code,
                    "error_code": request.state.error_code,
                }
                REQUEST_LOGGER.info(
                    json.dumps(
                        event,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
            end_request_timings(token)

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        request.state.request_outcome = "error"
        request.state.error_code = exc.code
        app.state.metrics.rag_request_errors.labels(code=exc.code).inc()
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(
                request,
                code=exc.code,
                message=exc.message,
                retryable=exc.retryable,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        del exc
        code = "INVALID_REQUEST"
        request.state.request_outcome = "error"
        request.state.error_code = code
        app.state.metrics.rag_request_errors.labels(code=code).inc()
        return JSONResponse(
            status_code=422,
            content=_error_body(
                request,
                code=code,
                message="Request không đúng schema",
                retryable=False,
            ),
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        code = "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR"
        request.state.request_outcome = "error"
        request.state.error_code = code
        app.state.metrics.rag_request_errors.labels(code=code).inc()
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(
                request,
                code=code,
                message=(
                    "Endpoint không tồn tại"
                    if exc.status_code == 404
                    else str(exc.detail)
                ),
                retryable=False,
            ),
        )

    @app.exception_handler(QdrantUnavailableError)
    async def qdrant_error_handler(
        request: Request,
        exc: QdrantUnavailableError,
    ) -> JSONResponse:
        code = "RETRIEVAL_UNAVAILABLE"
        request.state.request_outcome = "error"
        request.state.error_code = code
        app.state.metrics.rag_request_errors.labels(code=code).inc()
        LOGGER.warning(
            "Qdrant unavailable request_id=%s operation=%s",
            request.state.request_id,
            exc.operation,
        )
        return JSONResponse(
            status_code=503,
            content=_error_body(
                request,
                code=code,
                message="Dịch vụ truy xuất tạm thời không khả dụng",
                retryable=True,
            ),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        code = "INTERNAL_ERROR"
        request.state.request_outcome = "error"
        request.state.error_code = code
        app.state.metrics.rag_request_errors.labels(code=code).inc()
        LOGGER.exception("Unhandled API error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=_error_body(
                request,
                code=code,
                message="Lỗi nội bộ của dịch vụ",
                retryable=True,
            ),
        )

    app.include_router(chat.router)
    app.include_router(retrieve.router)
    app.include_router(system.router)
    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the legal RAG FastAPI API")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runtime.yaml"),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Load local secrets/settings from this dotenv file if it exists.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    load_environment_file(args.env_file)

    if args.require_cuda:
        os.environ["LEGAL_RAG_REQUIRE_CUDA"] = "true"
    os.environ["LEGAL_RAG_CONFIG"] = str(args.config)
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG

    log_level = os.environ.get("LEGAL_RAG_LOG_LEVEL", "INFO").upper()
    if log_level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        raise ValueError("LEGAL_RAG_LOG_LEVEL is invalid")
    log_config = copy.deepcopy(LOGGING_CONFIG)
    log_config["formatters"]["request_json"] = {"format": "%(message)s"}
    log_config["handlers"]["request_json"] = {
        "class": "logging.StreamHandler",
        "formatter": "request_json",
        "stream": "ext://sys.stdout",
    }
    log_config["loggers"]["legal_ir_prototype"] = {
        "handlers": ["default"],
        "level": log_level,
        "propagate": False,
    }
    log_config["loggers"]["legal_ir_prototype.api.requests"] = {
        "handlers": ["request_json"],
        "level": log_level,
        "propagate": False,
    }

    uvicorn.run(
        "legal_ir_prototype.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_config=log_config,
        access_log=False,
    )
