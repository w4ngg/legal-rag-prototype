from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping


PAYLOAD_SCHEMA_VERSION = 1
POINT_ID_NAMESPACE = uuid.UUID("1ab0f794-e98c-40fe-a2a7-10663c936801")

PAYLOAD_SCHEMA: dict[str, Any] = {
    "version": PAYLOAD_SCHEMA_VERSION,
    "required": {
        "schema_version": "integer",
        "chunk_id": "keyword",
        "document_id": "keyword",
        "vector_row": "integer",
        "granularity": "keyword",
        "passage": "text",
        "retrieval_text": "text",
        "passage_sha256": "keyword",
        "embedding_text_sha256": "keyword",
    },
    "optional": {
        "document_name": "text",
        "retrieval_title": "text",
        "chunk_index": "integer",
        "chunk_count_in_document": "integer",
        "character_start": "integer",
        "character_end": "integer",
        "source_file": "keyword",
        "source_link": "keyword",
        "chunk_strategy": "keyword",
        "normalization_version": "keyword",
    },
    "legal": {
        "document_type": "keyword|null",
        "document_number": "keyword|null",
        "issuing_authority": "keyword|null",
        "issued_at": "datetime|null",
        "effective_at": "datetime|null",
        "chapters": "keyword[]",
        "sections": "keyword[]",
        "articles": "keyword[]",
        "clauses": "keyword[]",
        "points": "keyword[]",
        "references": "object[]",
        "reference_keys": "keyword[]",
    },
}


def required_text(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point_id_for_chunk(chunk_id: str) -> str:
    return str(
        uuid.uuid5(
            POINT_ID_NAMESPACE,
            f"dsc-legal/chunk/{required_text(chunk_id, 'chunk_id')}",
        )
    )


def normalize_chunk_record(record: Mapping[str, Any]) -> dict[str, Any]:
    passage = required_text(record.get("passage", record.get("text")), "passage")
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object")
    return {
        "chunk_id": required_text(record.get("chunk_id"), "chunk_id"),
        "document_id": required_text(record.get("document_id"), "document_id"),
        "passage": passage,
        "retrieval_text": optional_text(record.get("retrieval_text")) or passage,
        "metadata": dict(metadata),
    }


def iter_chunk_records(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record must be an object")
                normalize_chunk_record(value)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid chunk at {source}:{line_number}: {exc}") from exc
            yield value


def canonical_chunk_hash(path: str | Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for record in iter_chunk_records(path):
        value = normalize_chunk_record(record)
        canonical = json.dumps(
            {
                "chunk_id": value["chunk_id"],
                "document_id": value["document_id"],
                "index_text": value["retrieval_text"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
        count += 1
    return digest.hexdigest(), count


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    source = [value] if isinstance(value, str) else value
    if not isinstance(source, (list, tuple, set)):
        return []
    result: list[str] = []
    for item in source:
        normalized = optional_text(item)
        if normalized is not None and normalized not in result:
            result.append(normalized)
    return result


def _legal_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    legal_source = metadata.get("legal") or {}
    if not isinstance(legal_source, Mapping):
        legal_source = {}

    def first(*keys: str) -> Any:
        for key in keys:
            if key in legal_source:
                return legal_source[key]
            if key in metadata:
                return metadata[key]
        return None

    legal: dict[str, Any] = {}
    scalar_fields = {
        "document_type": ("document_type", "legal_document_type"),
        "document_number": ("document_number", "legal_document_number"),
        "issuing_authority": ("issuing_authority",),
        "issued_at": ("issued_at", "issue_date"),
        "effective_at": ("effective_at", "effective_date"),
    }
    for output_name, input_names in scalar_fields.items():
        value = optional_text(first(*input_names))
        if value is not None:
            legal[output_name] = value
    list_fields = {
        "chapters": ("chapters", "chapter"),
        "sections": ("sections", "section"),
        "articles": ("articles", "article"),
        "clauses": ("clauses", "clause"),
        "points": ("points", "point"),
        "reference_keys": ("reference_keys", "legal_ref_keys"),
    }
    for output_name, input_names in list_fields.items():
        values = _string_list(first(*input_names))
        if values:
            legal[output_name] = values
    references = first("references", "legal_refs")
    if isinstance(references, list):
        valid = [item for item in references if isinstance(item, dict)]
        if valid:
            legal["references"] = valid
    return legal


def build_payload(
    record: Mapping[str, Any],
    *,
    vector_row: int,
    expected_granularity: str = "long",
) -> dict[str, Any]:
    normalized = normalize_chunk_record(record)
    metadata = normalized["metadata"]
    granularity = optional_text(metadata.get("granularity")) or expected_granularity
    if granularity != expected_granularity:
        raise ValueError(
            f"chunk {normalized['chunk_id']} has granularity={granularity!r}; "
            f"expected {expected_granularity!r}"
        )
    payload: dict[str, Any] = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "chunk_id": normalized["chunk_id"],
        "document_id": normalized["document_id"],
        "vector_row": int(vector_row),
        "granularity": granularity,
        "passage": normalized["passage"],
        "retrieval_text": normalized["retrieval_text"],
        "passage_sha256": hashlib.sha256(
            normalized["passage"].encode("utf-8")
        ).hexdigest(),
        "embedding_text_sha256": hashlib.sha256(
            normalized["retrieval_text"].encode("utf-8")
        ).hexdigest(),
        "legal": _legal_payload(metadata),
    }
    optional_fields = {
        "document_name": metadata.get("document_name"),
        "retrieval_title": metadata.get("retrieval_title"),
        "chunk_index": metadata.get("chunk_index"),
        "chunk_count_in_document": metadata.get("chunk_count_in_document"),
        "character_start": metadata.get("normalized_character_start"),
        "character_end": metadata.get("normalized_character_end"),
        "source_file": metadata.get("source_file"),
        "source_link": metadata.get("source_link"),
        "chunk_strategy": metadata.get("chunk_strategy"),
        "normalization_version": metadata.get("normalization_version"),
    }
    payload.update(
        {
            key: value
            for key, value in optional_fields.items()
            if value is not None and value != ""
        }
    )
    return payload
