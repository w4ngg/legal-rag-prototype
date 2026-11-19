from __future__ import annotations

"""Append-only, resumable artifact writer for evaluation runs."""

import json
import os
import fcntl
import logging
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel

from .schemas import EvaluationError, EvaluationPrediction


def read_jsonl(path: Path, *, repair_tail: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        offset = 0
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                offset += len(line)
                continue
            try:
                value = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                if not repair_tail or line.endswith(b"\n") or handle.read(1):
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
                backup = path.with_name(f"{path.name}.interrupted-{uuid.uuid4().hex}.tail")
                backup.write_bytes(line)
                with path.open("r+b") as writer:
                    writer.truncate(offset)
                    writer.flush()
                    os.fsync(writer.fileno())
                logging.getLogger(__name__).warning("Recovered interrupted JSONL tail: %s (backup %s)", path, backup)
                break
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            values.append(value)
            offset += len(line)
            if repair_tail and not line.endswith(b"\n"):
                with path.open("ab") as writer:
                    writer.write(b"\n")
                    writer.flush()
                    os.fsync(writer.fileno())
    return values


def latest_by_case(
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the newest append-only attempt for every case ID."""

    latest: dict[str, dict[str, Any]] = {}
    for value in values:
        case_id = str(value.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("artifact record is missing case_id")
        latest[case_id] = dict(value)
    return list(latest.values())


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: BaseModel | Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def artifact_lock(path: Path):
    """Prevent two processes from resuming the same output simultaneously."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another evaluation process owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ArtifactRecorder:
    def __init__(
        self,
        output_dir: Path,
        *,
        resume: bool,
        retry_errors: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.predictions_path = output_dir / "predictions.jsonl"
        self.errors_path = output_dir / "errors.jsonl"
        self.manifest_path = output_dir / "run_manifest.json"
        self.summary_path = output_dir / "summary.json"
        existing = tuple(
            path
            for path in (
                self.predictions_path,
                self.errors_path,
                self.manifest_path,
                self.summary_path,
            )
            if path.exists()
        )
        if existing and not resume:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(
                f"evaluation output already contains {names}; use --resume"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        prediction_rows = read_jsonl(self.predictions_path, repair_tail=resume)
        error_rows = read_jsonl(self.errors_path, repair_tail=resume)
        predictions = latest_by_case(prediction_rows)
        self._predictions = {str(value["case_id"]): value for value in predictions}
        self._errors = {str(value["case_id"]): value for value in latest_by_case(error_rows)}
        self._attempt_counts = Counter(str(value["case_id"]) for value in prediction_rows + error_rows)
        prediction_ids = {str(value["case_id"]) for value in predictions}
        effective_errors = [
            value
            for value in self._errors.values()
            if str(value["case_id"]) not in prediction_ids
        ]
        completed = set(prediction_ids)
        if not retry_errors:
            completed.update(str(value["case_id"]) for value in effective_errors)
        self.completed_case_ids = completed

    def write_manifest(self, value: Mapping[str, Any]) -> None:
        atomic_write_json(self.manifest_path, value)

    def write_prediction(self, value: EvaluationPrediction) -> None:
        self._ensure_new(value.case_id)
        append_jsonl(self.predictions_path, value)
        self._predictions[value.case_id] = value.model_dump(mode="json")
        self._attempt_counts[value.case_id] += 1
        self.completed_case_ids.add(value.case_id)

    def write_error(self, value: EvaluationError) -> None:
        self._ensure_new(value.case_id)
        append_jsonl(self.errors_path, value)
        self._errors[value.case_id] = value.model_dump(mode="json")
        self._attempt_counts[value.case_id] += 1
        self.completed_case_ids.add(value.case_id)

    def write_summary(self, value: Mapping[str, Any]) -> None:
        atomic_write_json(self.summary_path, value)

    def predictions(self) -> list[dict[str, Any]]:
        return list(self._predictions.values())

    def errors(self) -> list[dict[str, Any]]:
        prediction_ids = {
            str(value["case_id"]) for value in self.predictions()
        }
        return [
            value
            for value in self._errors.values()
            if str(value["case_id"]) not in prediction_ids
        ]

    def attempt_number(self, case_id: str) -> int:
        return self._attempt_counts[case_id] + 1

    def _ensure_new(self, case_id: str) -> None:
        if case_id in self.completed_case_ids:
            raise ValueError(f"case already completed: {case_id}")
