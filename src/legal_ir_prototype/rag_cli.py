from __future__ import annotations

"""Command-line smoke test for one complete grounded RAG request."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from .core import AppConfig
from .domain import LLMAnswer
from .llm import FakeLLMClient, LLMClient, OpenAICompatibleLLMClient
from .rag import (
    ContextBuildConfig,
    ContextBuilder,
    EvidenceChecker,
    EvidencePolicy,
    RAGExecution,
    RAGService,
)
from .services import build_service


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-rag-smoke",
        description=(
            "Run one end-to-end legal RAG query: retrieve, rerank, build "
            "context, call the LLM, and validate citations/evidence."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--show-context",
        action="store_true",
        help=(
            "Include selected/excluded context sources and the exact LLM "
            "system/user prompts in the JSON output."
        ),
    )
    return parser


def _fake_client(model: str) -> FakeLLMClient:
    return FakeLLMClient(
        LLMAnswer(
            answer="Đây là câu trả lời giả lập dựa trên nguồn được truy xuất.",
            citation_ids=("SOURCE_1",),
            model=model,
            confidence="high",
        )
    )


async def _answer_with_client(
    *,
    config: AppConfig,
    query: str,
    llm_client: LLMClient,
) -> RAGExecution:
    retrieval_service = build_service(config)
    rag_config = config.rag
    evidence_config = rag_config.evidence
    rag_service = RAGService(
        retrieval_service=retrieval_service,
        llm_client=llm_client,
        context_builder=ContextBuilder(
            ContextBuildConfig(
                max_context_tokens=rag_config.max_context_tokens,
                max_chunks=rag_config.max_chunks,
                max_chunks_per_document=rag_config.max_chunks_per_document,
                include_legal_metadata=rag_config.include_legal_metadata,
                merge_adjacent_chunks=rag_config.merge_adjacent_chunks,
            )
        ),
        evidence_checker=EvidenceChecker(
            EvidencePolicy(
                min_sources=evidence_config.min_sources,
                min_top_reranker_score=evidence_config.min_top_reranker_score,
                min_top_score_margin=evidence_config.min_top_score_margin,
                abstention_answer=evidence_config.abstention_answer,
            )
        ),
    )
    return await rag_service.execute(query)


def _result_payload(
    execution: RAGExecution,
    *,
    show_context: bool,
) -> dict[str, object]:
    result = execution.response.to_dict()
    if not show_context:
        return result
    prompt = execution.prompt
    result["debug"] = {
        "llm_called": prompt is not None,
        "context": execution.context.to_dict(),
        "llm_prompt": (
            None
            if prompt is None
            else {
                "system_prompt": prompt.system_prompt,
                "user_prompt": prompt.user_prompt,
            }
        ),
    }
    return result


async def _run(
    *,
    config_path: Path,
    query: str,
    show_context: bool,
) -> dict[str, object]:
    config = AppConfig.from_yaml(config_path)
    if config.llm is None:
        raise SystemExit(
            "configuration must include an llm section for legal-rag-smoke"
        )

    if config.llm.provider == "fake":
        execution = await _answer_with_client(
            config=config,
            query=query,
            llm_client=_fake_client(config.llm.model),
        )
    else:
        async with OpenAICompatibleLLMClient(config.llm) as llm_client:
            execution = await _answer_with_client(
                config=config,
                query=query,
                llm_client=llm_client,
            )
    return _result_payload(execution, show_context=show_context)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = asyncio.run(
        _run(
            config_path=args.config,
            query=args.query,
            show_context=args.show_context,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
