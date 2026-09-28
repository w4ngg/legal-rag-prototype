from __future__ import annotations

"""Public grounded legal-chat route."""

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, Depends, Request

from ...domain import Citation
from ...llm import LLMResponseError, LLMTransportError
from ..dependencies import get_metrics, get_request_id, get_runtime
from ..errors import APIError
from ..metrics import APIMetrics
from ..runtime import ApplicationRuntime
from ..schemas import ChatRequest, ChatResponse, CitationResponse


router = APIRouter(prefix="/v1", tags=["rag"])
LOGGER = logging.getLogger(__name__)


def _display(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return "; ".join(items) or None
    value = str(value).strip()
    return value or None


def _citation_response(citation: Citation) -> CitationResponse:
    metadata = citation.metadata
    legal = metadata.get("legal")
    legal = legal if isinstance(legal, Mapping) else {}
    return CitationResponse(
        source_id=citation.source_id,
        document_id=citation.document_id,
        document_name=_display(metadata.get("document_name")),
        article=_display(legal.get("articles")),
        source_url=_display(
            metadata.get("source_link")
            or metadata.get("source_url")
            or metadata.get("source_uri")
        ),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    runtime: ApplicationRuntime = Depends(get_runtime),
    metrics: APIMetrics = Depends(get_metrics),
    request_id: str = Depends(get_request_id),
) -> ChatResponse:
    try:
        async with request.app.state.inference_semaphore:
            execution = await runtime.rag_service.execute(payload.question)
    except LLMTransportError as exc:
        LOGGER.warning(
            "LLM transport failure request_id=%s error_type=%s detail=%s",
            request_id,
            type(exc).__name__,
            exc,
        )
        raise APIError(
            status_code=502,
            code="LLM_UNAVAILABLE",
            message="Không thể kết nối dịch vụ sinh câu trả lời",
            retryable=True,
        ) from exc
    except LLMResponseError as exc:
        LOGGER.warning(
            "LLM response rejected request_id=%s error_type=%s detail=%s",
            request_id,
            type(exc).__name__,
            exc,
        )
        raise APIError(
            status_code=502,
            code="INVALID_LLM_RESPONSE",
            message="Dịch vụ sinh câu trả lời trả về dữ liệu không hợp lệ",
            retryable=False,
        ) from exc

    metrics.observe_rag(execution)
    response = execution.response
    request.state.request_outcome = (
        "abstained" if response.abstained else "success"
    )
    return ChatResponse(
        answer=response.answer,
        citations=[_citation_response(value) for value in response.citations],
        sufficient_evidence=not response.abstained,
        abstention_reason=response.abstention_reason,
        warnings=list(response.warnings),
        request_id=request_id,
    )
