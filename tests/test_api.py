from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from fastapi.testclient import TestClient

from legal_ir_prototype.api.app import create_app
from legal_ir_prototype.api.runtime import ReadinessReport
from legal_ir_prototype.domain import LLMAnswer, LegalChunk
from legal_ir_prototype.llm import FakeLLMClient
from legal_ir_prototype.rag import RAGService
from legal_ir_prototype.vectorstore import QdrantUnavailableError


class FakeRetrieval:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[LegalChunk]:
        self.queries.append(query)
        return [
            LegalChunk(
                chunk_id="chunk-1",
                document_id="BLLD_2019",
                passage="Điều 41 quy định nghĩa vụ của người sử dụng lao động.",
                retrieval_text=(
                    "Điều 41. Nghĩa vụ khi đơn phương chấm dứt trái pháp luật."
                ),
                metadata={
                    "document_name": "Bộ luật Lao động 2019",
                    "source_link": "https://example.test/bbld-2019",
                    "legal": {"articles": ["Điều 41"]},
                },
                retrieval_scores={"dense": 0.8, "sparse": 12.0},
                retrieval_ranks={"dense": 1, "sparse": 2},
                fusion_score=0.03,
                fusion_rank=1,
                reranker_score=4.2,
                reranker_rank=1,
            )
        ]


class FakeRuntime:
    def __init__(self, *, ready: bool = True) -> None:
        self.retrieval_service = FakeRetrieval()
        self.llm_client = FakeLLMClient(
            LLMAnswer(
                answer=(
                    "Theo Điều 41, người sử dụng lao động phải thực hiện nghĩa vụ."
                ),
                citation_ids=("SOURCE_1",),
                model="fake-model",
            )
        )
        self.rag_service = RAGService(
            retrieval_service=self.retrieval_service,
            llm_client=self.llm_client,
        )
        self.config = SimpleNamespace(
            reranker=SimpleNamespace(output_top_k=20)
        )
        self._ready = ready
        self.closed = False

    def startup(self) -> None:
        return None

    def readiness(self) -> ReadinessReport:
        return ReadinessReport(
            embedding=self._ready,
            reranker=self._ready,
            qdrant=self._ready,
            cuda=True,
            llm=self._ready,
        )

    async def aclose(self) -> None:
        self.closed = True

    def version_info(self) -> dict[str, object]:
        return {
            "service_version": "0.1.0",
            "release_id": "test",
            "git_sha": "abc123",
            "dense_model": "dense",
            "dense_revision": "1" * 40,
            "reranker_model": "reranker",
            "reranker_revision": "2" * 40,
            "qdrant_collection": "legal",
            "llm_model": "fake-model",
            "prompt_version": "legal_qa_v2",
            "output_schema_version": 1,
        }


def _client(
    *,
    ready: bool = True,
    retrieve: bool = True,
    cors_origins: tuple[str, ...] = (),
) -> TestClient:
    return TestClient(
        create_app(
            runtime=FakeRuntime(ready=ready),
            enable_retrieve_endpoint=retrieve,
            internal_api_key="test-secret" if retrieve else None,
            cors_origins=cors_origins,
        )
    )


def test_health_is_a_dependency_free_liveness_check() -> None:
    app = create_app(
        config_path="/path/that/does/not/exist.yaml",
        enable_retrieve_endpoint=False,
    )
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_200_or_503_from_dependency_state() -> None:
    with _client(ready=True) as client:
        ready = client.get("/ready")
    with _client(ready=False) as client:
        not_ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert not_ready.status_code == 503
    assert not_ready.json()["qdrant"] is False


def test_chat_returns_validated_structured_response_and_request_id() -> None:
    with _client() as client:
        response = client.post(
            "/v1/chat",
            json={"question": "Sa thải trái pháp luật xử lý thế nào?"},
            headers={"X-Request-ID": "abc-123"},
        )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "abc-123"
    body = response.json()
    assert body["request_id"] == "abc-123"
    assert body["sufficient_evidence"] is True
    assert body["citations"] == [
        {
            "source_id": "SOURCE_1",
            "document_id": "BLLD_2019",
            "document_name": "Bộ luật Lao động 2019",
            "article": "Điều 41",
            "source_url": "https://example.test/bbld-2019",
        }
    ]


def test_chat_cors_preflight_allows_only_configured_frontend() -> None:
    with _client(cors_origins=("http://localhost:3000",)) as client:
        allowed = client.options(
            "/v1/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-request-id",
            },
        )
        denied = client.options(
            "/v1/chat",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == (
        "http://localhost:3000"
    )
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers


def test_retrieve_requires_internal_key_and_returns_scores() -> None:
    with _client() as client:
        denied = client.post("/v1/retrieve", json={"query": "Điều 41"})
        allowed = client.post(
            "/v1/retrieve",
            json={"query": "Điều 41", "top_k": 1},
            headers={"X-Internal-API-Key": "test-secret"},
        )
    assert denied.status_code == 401
    assert allowed.status_code == 200
    result = allowed.json()["results"][0]
    assert result["chunk_id"] == "chunk-1"
    assert result["reranker_score"] == 4.2
    assert result["fusion_score"] == 0.03


def test_retrieve_is_disabled_by_default() -> None:
    with _client(retrieve=False) as client:
        response = client.post("/v1/retrieve", json={"query": "Điều 41"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_metrics_version_and_validation_contracts() -> None:
    with _client() as client:
        client.post("/v1/chat", json={"question": "Một câu hỏi"})
        metrics = client.get("/metrics")
        version = client.get("/version")
        invalid = client.post("/v1/chat", json={"question": " "})
    assert metrics.status_code == 200
    assert "rag_requests_total" in metrics.text
    assert "retrieval_duration_seconds" in metrics.text
    assert "reranker_duration_seconds" in metrics.text
    assert "llm_duration_seconds" in metrics.text
    assert "qdrant_request_duration_seconds" in metrics.text
    assert version.status_code == 200
    assert version.json()["prompt_version"] == "legal_qa_v2"
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "INVALID_REQUEST"


def test_request_completion_log_is_structured_and_correlated(
    caplog,
) -> None:
    caplog.set_level(
        logging.INFO,
        logger="legal_ir_prototype.api.requests",
    )
    with _client() as client:
        response = client.post(
            "/v1/chat",
            json={"question": "Một câu hỏi"},
            headers={"X-Request-ID": "trace-123"},
        )

    records = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "legal_ir_prototype.api.requests"
    ]
    assert response.status_code == 200
    assert len(records) == 1
    event = records[0]
    assert event["event"] == "http_request_completed"
    assert event["request_id"] == "trace-123"
    assert event["status"] == "success"
    assert event["http_status"] == 200
    assert event["total_ms"] >= 0
    assert "retrieval_ms" in event
    assert "reranker_ms" in event
    assert "context_builder_ms" in event
    assert "llm_ms" in event
    serialized = json.dumps(event, ensure_ascii=False)
    assert "Một câu hỏi" not in serialized
    assert "Câu trả lời" not in serialized
    assert "test-secret" not in serialized


def test_request_completion_log_is_emitted_only_for_chat(caplog) -> None:
    caplog.set_level(
        logging.INFO,
        logger="legal_ir_prototype.api.requests",
    )
    with _client() as client:
        health = client.get("/health")
        ready = client.get("/ready")
        metrics = client.get("/metrics")
        version = client.get("/version")
        retrieve = client.post(
            "/v1/retrieve",
            json={"query": "Điều 41", "top_k": 1},
            headers={"X-Internal-API-Key": "test-secret"},
        )

    records = [
        record
        for record in caplog.records
        if record.name == "legal_ir_prototype.api.requests"
    ]
    assert health.status_code == 200
    assert ready.status_code == 200
    assert metrics.status_code == 200
    assert version.status_code == 200
    assert retrieve.status_code == 200
    assert records == []


def test_qdrant_failure_maps_to_controlled_503_response() -> None:
    class FailingRetrieval:
        def retrieve(self, query: str):
            del query
            raise QdrantUnavailableError("query_points:dense")

    runtime = FakeRuntime()
    runtime.rag_service = RAGService(
        retrieval_service=FailingRetrieval(),
        llm_client=runtime.llm_client,
    )
    app = create_app(
        runtime=runtime,
        enable_retrieve_endpoint=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={"question": "Một câu hỏi"},
            headers={"X-Request-ID": "qdrant-error-1"},
        )

    assert response.status_code == 503
    assert response.headers["X-Request-ID"] == "qdrant-error-1"
    assert response.json() == {
        "request_id": "qdrant-error-1",
        "error": {
            "code": "RETRIEVAL_UNAVAILABLE",
            "message": "Dịch vụ truy xuất tạm thời không khả dụng",
            "retryable": True,
        },
    }
