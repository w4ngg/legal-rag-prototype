from __future__ import annotations

"""Create and load frozen QA evaluation datasets."""

import hashlib
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schemas import DatasetManifest, EvalCase, ExcludedSourceCase


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, values: list[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, path)


def _source_rows(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid source JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("source QA dataset must be a JSON object keyed by query ID")
    return {str(key): row for key, row in value.items()}


def question_key(question: str) -> str:
    return " ".join(unicodedata.normalize("NFC", question).casefold().split())


def _tags(question: str, answer: str) -> tuple[str, ...]:
    temporal = re.search(r"năm\s+20\d\d|hiện nay|hiện hành|hiệu lực|thời điểm|trước ngày|sau ngày", question, re.I)
    articles = set(re.findall(r"điều\s+\d+", answer, re.I))
    slice_name = "temporal" if temporal else "multi_reference" if len(articles) >= 2 else "standard"
    topics = {
        "labor": r"lao động|tiền lương|nghỉ phép|bảo hiểm xã hội|sa thải|việc làm",
        "land_housing": r"đất|nhà ở|bất động sản|xây dựng|chung cư",
        "family_civil": r"kết hôn|ly hôn|thừa kế|con nuôi|giám hộ|hợp đồng|bồi thường",
        "business_tax": r"thuế|doanh nghiệp|kinh doanh|công ty|hóa đơn|ngân hàng",
        "criminal_procedure": r"hình sự|phạm tội|bị can|bị cáo|tố tụng|khởi kiện",
        "transport": r"giao thông|lái xe|đăng kiểm|đăng ký xe|đường bộ",
        "education_health": r"học sinh|sinh viên|giáo dục|trường học|bệnh viện|khám bệnh|y tế",
    }
    topic = next((name for name, pattern in topics.items() if re.search(pattern, question, re.I)), "administrative_other")
    return ("qa_reference", f"slice:{slice_name}", f"topic:{topic}")


def write_dataset(*, cases: Sequence[EvalCase], output_dir: Path,
                  selection_method: str, sources: Sequence[Mapping[str, Any]],
                  seed: int = 2026, excluded: Sequence[ExcludedSourceCase] = (),
                  overwrite: bool = False) -> DatasetManifest:
    cases_path, manifest_path = output_dir / "cases.jsonl", output_dir / "manifest.json"
    if not overwrite and (cases_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"frozen dataset already exists in {output_dir}; use --overwrite explicitly")
    if not cases or len({c.case_id for c in cases}) != len(cases):
        raise ValueError("dataset must contain unique non-empty case IDs")
    if len({question_key(c.question) for c in cases}) != len(cases):
        raise ValueError("dataset contains duplicate normalized questions")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(cases_path, [case.model_dump(mode="json") for case in cases])
    primary = sources[0]
    manifest = DatasetManifest(
        dataset_id=output_dir.name, created_at=datetime.now(timezone.utc).isoformat(),
        source_path=str(primary["path"]), source_sha256=str(primary["sha256"]),
        selection_method=selection_method, seed=seed,
        requested_sample_size=len(cases), selected_sample_size=len(cases),
        selected_source_query_ids=tuple(c.source_query_id for c in cases),
        excluded_cases=tuple(excluded), sources=tuple(dict(s) for s in sources),
        slice_counts=dict(Counter(tag for c in cases for tag in c.tags)),
        cases_sha256=sha256_file(cases_path),
    )
    _atomic_json(manifest_path, manifest.model_dump(mode="json"))
    return manifest


def prepare_dataset(
    *,
    source: Path | Sequence[Path],
    output_dir: Path,
    sample_size: int,
    seed: int,
    overwrite: bool = False,
    strategy: str = "random",
    source_counts: Sequence[int] | None = None,
    exclude_datasets: Sequence[Path] = (),
) -> DatasetManifest:
    paths = [source] if isinstance(source, Path) else list(source)
    paths = [path.resolve() for path in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("provide distinct source QA datasets")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"source QA dataset does not exist: {path}")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    cases_path = output_dir / "cases.jsonl"
    manifest_path = output_dir / "manifest.json"
    if not overwrite and (cases_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"frozen dataset already exists in {output_dir}; use --overwrite explicitly"
        )

    if strategy not in {"random", "stratified"}:
        raise ValueError("strategy must be random or stratified")
    if source_counts is not None and (len(source_counts) != len(paths)
            or sum(source_counts) != sample_size or any(n < 0 for n in source_counts)):
        raise ValueError("source_counts must match sources and sum to sample_size")
    seen = {question_key(case.question) for path in exclude_datasets for case in load_cases(path)}
    cases: list[EvalCase] = []
    excluded: list[ExcludedSourceCase] = []
    sources = []
    rng = random.Random(seed)
    for source_index, path in enumerate(paths):
        rows = _source_rows(path)
        ids = sorted(rows)
        rng.shuffle(ids)
        candidates = []
        for query_id in ids:
            row = rows[query_id]
            if (not isinstance(row, dict) or not isinstance(row.get("question"), str)
                    or not isinstance(row.get("answer"), str)
                    or not row["question"].strip() or not row["answer"].strip()):
                excluded.append(ExcludedSourceCase(source_query_id=query_id, reason=f"{path.name}: missing_labeled_question_or_answer"))
                continue
            key = question_key(row["question"])
            if key in seen:
                excluded.append(ExcludedSourceCase(source_query_id=query_id, reason=f"{path.name}: duplicate_or_excluded_question"))
                continue
            seen.add(key)
            candidates.append(EvalCase(
                case_id=f"qa_{path.stem}_{query_id}" if len(paths) > 1 else f"qa_{query_id}",
                source_query_id=query_id, question=row["question"], reference_answer=row["answer"],
                source_file=f"{path.parent.name}/{path.name}",
                tags=(*_tags(row["question"], row["answer"]), f"source:{path.stem}"),
            ))
        if strategy == "stratified":
            buckets: dict[tuple[str, ...], deque[EvalCase]] = defaultdict(deque)
            for case in candidates:
                buckets[tuple(t for t in case.tags if t.startswith(("slice:", "topic:")))].append(case)
            ordered = []
            keys = sorted(buckets)
            rng.shuffle(keys)
            while any(buckets.values()):
                for key in keys:
                    if buckets[key]:
                        ordered.append(buckets[key].popleft())
            candidates = ordered
        count = source_counts[source_index] if source_counts is not None else sample_size - len(cases)
        if len(candidates) < count and source_counts is not None:
            raise ValueError(f"only {len(candidates)} unique labeled rows available in {path}; requested {count}")
        selected = candidates[:count]
        cases.extend(selected)
        sources.append({"path": str(path), "sha256": sha256_file(path),
                        "selected_count": len(selected), "source_row_count": len(rows)})
    if len(cases) != sample_size:
        raise ValueError(
            f"only {len(cases)} valid cases were available for sample_size={sample_size}"
        )

    return write_dataset(cases=cases, output_dir=output_dir, seed=seed, sources=sources,
                         excluded=excluded, overwrite=overwrite,
                         selection_method=("stratified_unique_questions" if strategy == "stratified"
                                           else "seeded_random_without_replacement"))


def load_cases(path: Path) -> tuple[EvalCase, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"evaluation dataset does not exist: {path}")
    cases: list[EvalCase] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = EvalCase.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"invalid eval case at {path}:{line_number}: {exc}") from exc
            if case.case_id in seen:
                raise ValueError(f"duplicate case_id in eval dataset: {case.case_id}")
            seen.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError(f"evaluation dataset is empty: {path}")
    return tuple(cases)
