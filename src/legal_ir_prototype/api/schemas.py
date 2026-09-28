from __future__ import annotations

"""Versioned HTTP request/response contracts."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4_000)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be empty")
        return normalized


class CitationResponse(BaseModel):
    source_id: str
    document_id: str
    document_name: str | None = None
    article: str | None = None
    source_url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    sufficient_evidence: bool
    abstention_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    request_id: str


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4_000)
    top_k: int = Field(default=5, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        return normalized


class RetrievalResultResponse(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    retrieval_scores: dict[str, float]
    retrieval_ranks: dict[str, int]
    fusion_score: float | None = None
    fusion_rank: int | None = None
    reranker_score: float | None = None
    reranker_rank: int | None = None


class RetrieveResponse(BaseModel):
    request_id: str
    result_count: int
    results: list[RetrievalResultResponse]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    embedding: bool
    reranker: bool
    qdrant: bool
    cuda: bool
    llm: bool


class VersionResponse(BaseModel):
    request_id: str
    service_version: str
    release_id: str
    git_sha: str | None
    dense_model: str
    dense_revision: str
    reranker_model: str
    reranker_revision: str
    qdrant_collection: str
    llm_model: str
    prompt_version: str
    output_schema_version: int


class APIErrorBody(BaseModel):
    code: str
    message: str
    retryable: bool


class APIErrorResponse(BaseModel):
    request_id: str
    error: APIErrorBody
