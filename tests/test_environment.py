from __future__ import annotations

import os

from legal_ir_prototype.environment import load_environment_file


def test_load_environment_file_without_overriding_process_environment(
    tmp_path,
    monkeypatch,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "LEGAL_RAG_LLM_API_KEY=file-secret\n"
        "LEGAL_RAG_CORS_ORIGINS=http://localhost:3000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LEGAL_RAG_LLM_API_KEY", "process-secret")
    monkeypatch.delenv("LEGAL_RAG_CORS_ORIGINS", raising=False)

    assert load_environment_file(dotenv_path) is True
    assert os.environ["LEGAL_RAG_LLM_API_KEY"] == "process-secret"
    assert os.environ["LEGAL_RAG_CORS_ORIGINS"] == (
        "http://localhost:3000"
    )


def test_missing_environment_file_is_optional(tmp_path) -> None:
    assert load_environment_file(tmp_path / "missing.env") is False
