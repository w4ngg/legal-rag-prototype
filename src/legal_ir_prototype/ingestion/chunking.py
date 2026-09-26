from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

from .config import ChunkingConfig


FORMAT_VERSION = 1
CHUNKING_VERSION = "dual_char_v1"
NORMALIZATION_VERSION = "legal_nfc_lines_v1"
LONG_CHUNK_STRATEGY = "overlapping_short_window_character"

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f-\x9f]")
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_WHITESPACE = re.compile(r"\s+")
_HTML_SPACE_ENTITY = re.compile(r"&(?:nbsp|#0*160|#x0*a0);", re.IGNORECASE)
_CONTEXT_FILENAME = re.compile(r"^context_(.+)$")
_INVISIBLE_TRANSLATION = str.maketrans(
    {
        "\u00ad": None,
        "\u200b": " ",
        "\u200c": None,
        "\u200d": None,
        "\u200e": None,
        "\u200f": None,
        "\u2060": None,
        "\ufeff": None,
        "\u2028": "\n",
        "\u2029": "\n",
    }
)
_STRUCTURAL_LINE_START = re.compile(
    r"^(?:"
    r"(?:phần|chương|mục|tiểu\s+mục|điều|khoản)\s+"
    r"(?:số\s+)?(?:\d+[a-zđ]?|[ivxlcdm]+)(?:[.):-]|\s|$)"
    r"|phụ\s+lục\s+\S+"
    r"|(?:\d+(?:\.\d+){0,4}|[a-zđ]|[ivxlcdm]+)[.)](?:\s|$)"
    r"|[-–—•](?:\s|$)"
    r")",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?;:][\"'”’)]*(?=\s|$)")


@dataclass(frozen=True, slots=True)
class TextSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid text span: {self.start}:{self.end}")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(slots=True)
class ChunkingStatistics:
    documents_discovered: int = 0
    documents_chunked: int = 0
    documents_skipped_empty: int = 0
    normalized_characters: int = 0
    long_chunks_written: int = 0
    maximum_long_chunk_characters: int = 0


def normalize_legal_text_with_lines(value: Any) -> str:
    if value is None:
        return ""
    text = _HTML_SPACE_ENTITY.sub(" ", str(value))
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLE_TRANSLATION)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\f", "\n").replace("\v", "\n")
    text = _CONTROL_CHARACTERS.sub(" ", text)
    lines = [
        _HORIZONTAL_WHITESPACE.sub(" ", line).strip()
        for line in text.split("\n")
    ]
    return "\n".join(line for line in lines if line)


def _single_line_text(value: Any) -> str:
    text = normalize_legal_text_with_lines(value)
    return _WHITESPACE.sub(" ", text).strip()


def _source_sort_key(path: Path) -> tuple[int, int | str, str]:
    match = _CONTEXT_FILENAME.match(path.stem)
    if match and match.group(1).isdigit():
        return (0, int(match.group(1)), path.name)
    return (1, path.name, path.name)


def discover_raw_documents(input_dir: Path, pattern: str) -> list[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"raw input is not a directory: {input_dir}")
    paths = sorted(input_dir.glob(pattern), key=_source_sort_key)
    if not paths:
        raise FileNotFoundError(
            f"no raw documents matched {pattern!r} under {input_dir}"
        )
    return paths


def read_raw_document(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read raw document {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"raw document must be an object: {path}")
    if value.get("id") is None or not str(value["id"]).strip():
        raise ValueError(f"raw document has no valid id: {path}")
    if not isinstance(value.get("passage"), str):
        raise ValueError(f"raw document passage must be a string: {path}")
    document_id = str(value["id"]).strip()
    match = _CONTEXT_FILENAME.match(path.stem)
    if match and match.group(1) != document_id:
        raise ValueError(
            f"filename/document id mismatch: {path.name} contains {document_id!r}"
        )
    return value, raw


def _trim_span(text: str, start: int, end: int) -> TextSpan | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return TextSpan(start, end) if start < end else None


def _line_spans(text: str) -> list[TextSpan]:
    return [TextSpan(match.start(), match.end()) for match in re.finditer(r"[^\n]+", text)]


def _logical_clause_spans(text: str) -> list[TextSpan]:
    lines = _line_spans(text)
    if not lines:
        return []
    result: list[TextSpan] = []
    unit_start = lines[0].start
    previous_end = lines[0].end
    for line in lines[1:]:
        if _STRUCTURAL_LINE_START.match(text[line.start : line.end]):
            span = _trim_span(text, unit_start, previous_end)
            if span is not None:
                result.append(span)
            unit_start = line.start
        previous_end = line.end
    span = _trim_span(text, unit_start, previous_end)
    if span is not None:
        result.append(span)
    return result


def _preferred_split(text: str, start: int, limit: int) -> int:
    minimum = start + max(1, (limit - start) // 2)
    region = text[start:limit]
    sentence_ends = [
        start + match.end()
        for match in _SENTENCE_END.finditer(region)
        if start + match.end() >= minimum
    ]
    if sentence_ends:
        return sentence_ends[-1]
    newline = text.rfind("\n", minimum, limit + 1)
    if newline >= minimum:
        return newline
    for position in range(limit, minimum - 1, -1):
        if position < len(text) and text[position].isspace():
            return position
    return limit


def _split_span(text: str, span: TextSpan, maximum: int) -> list[TextSpan]:
    pieces: list[TextSpan] = []
    cursor = span.start
    while cursor < span.end:
        while cursor < span.end and text[cursor].isspace():
            cursor += 1
        if cursor >= span.end:
            break
        limit = min(cursor + maximum, span.end)
        split = span.end if limit == span.end else _preferred_split(text, cursor, limit)
        if split <= cursor:
            split = limit
        piece = _trim_span(text, cursor, split)
        if piece is not None:
            pieces.append(piece)
        cursor = split
    return pieces


def short_chunk_spans(text: str, maximum: int) -> list[TextSpan]:
    chunks: list[TextSpan] = []
    for unit in _logical_clause_spans(text):
        chunks.extend(_split_span(text, unit, maximum))
    return chunks


def long_chunk_spans(
    short_spans: Sequence[TextSpan],
    *,
    maximum: int,
    overlap_short_chunks: int,
) -> list[TextSpan]:
    if not short_spans:
        return []
    if any(span.length > maximum for span in short_spans):
        raise ValueError("a short chunk is larger than the long chunk maximum")
    result: list[TextSpan] = []
    start_index = 0
    while start_index < len(short_spans):
        end_index = start_index + 1
        while end_index < len(short_spans):
            if short_spans[end_index].end - short_spans[start_index].start > maximum:
                break
            end_index += 1
        result.append(
            TextSpan(short_spans[start_index].start, short_spans[end_index - 1].end)
        )
        if end_index == len(short_spans):
            break
        start_index = max(start_index + 1, end_index - overlap_short_chunks)
    return result


def _document_titles(document: Mapping[str, Any], maximum: int) -> tuple[str, str]:
    original = _single_line_text(document.get("name"))
    if not original:
        link = str(document.get("link") or "").strip()
        if link:
            original = _single_line_text(
                Path(unquote(urlparse(link).path)).stem
            )
    retrieval = _WHITESPACE.sub(" ", re.sub(r"[-_]+", " ", original)).strip()
    if len(retrieval) > maximum:
        retrieval = retrieval[:maximum].rstrip()
    return original, retrieval


def _namespace(config: ChunkingConfig) -> str:
    return (
        f"{CHUNKING_VERSION}_{NORMALIZATION_VERSION}_"
        f"s{config.short_max_characters}_l{config.long_max_characters}_"
        f"o{config.long_overlap_short_chunks}"
    )


def build_document_long_chunks(
    document: Mapping[str, Any],
    source_path: Path,
    config: ChunkingConfig,
) -> list[dict[str, Any]]:
    document_id = str(document["id"]).strip()
    text = normalize_legal_text_with_lines(document.get("passage"))
    if not text:
        return []
    short_spans = short_chunk_spans(text, config.short_max_characters)
    long_spans = long_chunk_spans(
        short_spans,
        maximum=config.long_max_characters,
        overlap_short_chunks=config.long_overlap_short_chunks,
    )
    if not long_spans:
        raise ValueError(f"chunking produced no long chunks for {document_id}")
    namespace = _namespace(config)
    short_ids = [
        f"{document_id}:{namespace}:short:{index:06d}"
        for index in range(len(short_spans))
    ]
    long_ids = [
        f"{document_id}:{namespace}:long:{index:06d}"
        for index in range(len(long_spans))
    ]
    original_title, retrieval_title = _document_titles(
        document, config.title_max_characters
    )
    source_link = str(document.get("link") or "").strip()
    raw_metadata = document.get("metadata") or {}
    if not isinstance(raw_metadata, Mapping):
        raise ValueError(f"raw metadata must be an object: {source_path}")
    common: dict[str, Any] = {
        "normalization_version": NORMALIZATION_VERSION,
        "normalized_document_characters": len(text),
        "source_file": source_path.name,
    }
    if original_title:
        common["document_name"] = original_title
    if retrieval_title:
        common["retrieval_title"] = retrieval_title
    if source_link:
        common["source_link"] = source_link
    if isinstance(raw_metadata.get("legal"), Mapping):
        common["legal"] = dict(raw_metadata["legal"])

    records: list[dict[str, Any]] = []
    for index, span in enumerate(long_spans):
        passage = text[span.start : span.end]
        members = [
            short_ids[position]
            for position, short_span in enumerate(short_spans)
            if span.start <= short_span.start and short_span.end <= span.end
        ]
        metadata = {
            **common,
            "granularity": "long",
            "chunk_strategy": LONG_CHUNK_STRATEGY,
            "chunk_index": index,
            "chunk_count_in_document": len(long_spans),
            "max_characters": config.long_max_characters,
            "character_count": len(passage),
            "normalized_character_start": span.start,
            "normalized_character_end": span.end,
            "overlap_short_chunks": config.long_overlap_short_chunks,
            "short_chunk_ids": members,
        }
        retrieval_text = (
            f"Tên văn bản: {retrieval_title}\n{passage}"
            if config.include_document_title and retrieval_title
            else passage
        )
        records.append(
            {
                "chunk_id": long_ids[index],
                "document_id": document_id,
                "passage": passage,
                "metadata": metadata,
                "retrieval_text": retrieval_text,
            }
        )
    return records


def _serialized_line(record: Mapping[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def _temporary_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(descriptor)
    return Path(name)


def build_long_chunks(
    raw_dir: str | Path,
    output_dir: str | Path,
    config: ChunkingConfig,
    *,
    overwrite: bool = False,
) -> Path:
    source_root = Path(raw_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    chunks_path = destination / "long_chunks.jsonl"
    manifest_path = destination / "manifest.json"
    existing = [path for path in (chunks_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "chunk artifacts already exist; pass --overwrite explicitly: "
            + ", ".join(str(path) for path in existing)
        )
    destination.mkdir(parents=True, exist_ok=True)
    temporary_chunks = _temporary_path(chunks_path)
    temporary_manifest = _temporary_path(manifest_path)
    output_digest = hashlib.sha256()
    input_digest = hashlib.sha256()
    paths = discover_raw_documents(source_root, config.input_pattern)
    statistics = ChunkingStatistics(documents_discovered=len(paths))
    seen_ids: set[str] = set()
    try:
        with temporary_chunks.open("w", encoding="utf-8", newline="\n") as handle:
            for path in paths:
                document, raw = read_raw_document(path)
                document_id = str(document["id"]).strip()
                if document_id in seen_ids:
                    raise ValueError(f"duplicate document id: {document_id}")
                seen_ids.add(document_id)
                relative_name = path.relative_to(source_root).as_posix()
                input_digest.update(relative_name.encode("utf-8"))
                input_digest.update(b"\0")
                input_digest.update(raw)
                input_digest.update(b"\0")
                records = build_document_long_chunks(document, path, config)
                if not records:
                    statistics.documents_skipped_empty += 1
                    continue
                statistics.documents_chunked += 1
                normalized = normalize_legal_text_with_lines(document["passage"])
                statistics.normalized_characters += len(normalized)
                statistics.long_chunks_written += len(records)
                statistics.maximum_long_chunk_characters = max(
                    statistics.maximum_long_chunk_characters,
                    max(len(record["passage"]) for record in records),
                )
                for record in records:
                    line = _serialized_line(record)
                    handle.write(line)
                    output_digest.update(line.encode("utf-8"))
        if statistics.long_chunks_written == 0:
            raise ValueError("chunking produced no long chunks")
        manifest = {
            "format_version": FORMAT_VERSION,
            "artifact_type": "legal_rag_long_chunks",
            "strategy": "dual_character_long_only",
            "strategy_version": CHUNKING_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "input": {
                "directory": str(source_root),
                "pattern": config.input_pattern,
                "sha256": input_digest.hexdigest(),
            },
            "output": {
                "filename": chunks_path.name,
                "sha256": output_digest.hexdigest(),
                "records": statistics.long_chunks_written,
            },
            "chunking": asdict(config),
            "statistics": asdict(statistics),
        }
        with temporary_manifest.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary_chunks.replace(chunks_path)
        temporary_manifest.replace(manifest_path)
        return chunks_path
    except BaseException:
        temporary_chunks.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise
