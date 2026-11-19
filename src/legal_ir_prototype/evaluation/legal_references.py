from __future__ import annotations

"""Conservative legal-anchor extraction. Matches are weak labels, not IR gold."""

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


EXTRACTOR_VERSION = "legal_anchors_v1"


def fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
                   if unicodedata.category(c) != "Mn")


@dataclass(frozen=True)
class LegalReference:
    document: str | None
    article: str | None = None
    clause: str | None = None
    point: str | None = None
    raw: str = ""
    document_inferred_from_metadata: bool = False


_NUMBER = re.compile(r"\b(\d{1,4})\s*/\s*((?:19|20)\d{2})\s*/\s*([a-z]+(?:\s*-\s*[a-z0-9]+)*)\b")
_LAW = re.compile(r"\b((?:bo\s+)?luat\s+[a-z][a-z\s,]{1,85}?)\s+(?:nam\s+)?((?:19|20)\d{2})\b")
_SLUG_NUMBER = re.compile(r"\b(?:nghi-dinh|thong-tu|quyet-dinh|nghi-quyet)-(\d+)-((?:19|20)\d{2})-(nd-cp|tt-[a-z]+|qd-[a-z0-9]+|nq-[a-z0-9]+)\b")
_ARTICLE = re.compile(r"\bdieu\s+(\d+[a-z]?)\b")
_CLAUSE = re.compile(r"\bkhoan\s+(\d+[a-z]?)\b")
_POINT = re.compile(r"\bdiem\s+([a-z])\b")


def _fold_offsets(text: str) -> tuple[str, list[int]]:
    chars, offsets = [], []
    for index, char in enumerate(text):
        for value in fold(char):
            chars.append(value)
            offsets.append(index)
    return "".join(chars), offsets


def _document_mentions(text: str) -> list[tuple[int, int, str]]:
    found = []
    for pattern in (_NUMBER, _SLUG_NUMBER):
        for match in pattern.finditer(text):
            suffix = re.sub(r"\s+", "", match[3]).upper()
            key = f"{int(match[1])}/{match[2]}/{suffix}"
            found.append((match.start(), match.end(), key))
    for match in _LAW.finditer(text):
        name = " ".join(match[1].replace(",", " ").split())
        found.append((match.start(), match.end(), f"{name}:{match[2]}"))
    for match in re.finditer(r"\b((?:bo-)?luat-(?:[a-z]+-){1,16})((?:19|20)\d{2})\b", text):
        name = match[1].strip("-").replace("-", " ")
        found.append((match.start(), match.end(), f"{name}:{match[2]}"))
    return sorted(set(found))


def extract_references(text: str, *, document_hint: str | None = None) -> list[LegalReference]:
    normalized, offsets = _fold_offsets(text)
    documents = _document_mentions(normalized)
    result: list[LegalReference] = []
    used_documents: set[tuple[int, int, str]] = set()
    articles = list(_ARTICLE.finditer(normalized))
    for index, article in enumerate(articles):
        # Only associate explicit citations within the same line/sentence. An
        # unresolved article remains unresolved instead of borrowing a distant law.
        left = max(normalized.rfind("\n", 0, article.start()), normalized.rfind(";", 0, article.start()),
                   normalized.rfind(".", 0, article.start())) + 1
        ends = [p for marker in ("\n", ";", ".") if (p := normalized.find(marker, article.end())) >= 0]
        right = min(ends, default=len(normalized))
        candidates = [d for d in documents if d[0] >= left and d[1] <= right
                      and min(abs(d[0] - article.end()), abs(article.start() - d[1])) <= 180]
        following = []
        for candidate in candidates:
            if candidate[0] < article.end():
                continue
            gap = normalized[article.end():candidate[0]]
            if not _ARTICLE.search(gap) and (re.search(r"nghi dinh|thong tu|quyet dinh|nghi quyet", gap)
                                            or not gap.strip(" ,:()") or gap.strip(" ,:()") == "cua"):
                following.append(candidate)
        nearest = (min(following, key=lambda d: d[0]) if following else
                   min(candidates, key=lambda d: min(abs(d[0] - article.end()), abs(article.start() - d[1])), default=None))
        document = nearest[2] if nearest else document_hint
        if nearest:
            used_documents.add(nearest)
        previous_article_end = articles[index - 1].end() if index else left
        next_article_start = articles[index + 1].start() if index + 1 < len(articles) else right
        before = normalized[max(left, previous_article_end, article.start() - 55):article.start()]
        after = normalized[article.end():min(right, next_article_start, article.end() + 55)]
        # A preceding clause belongs to this article. A following clause is only
        # accepted when it immediately follows the article, before a law name.
        before_clauses = list(_CLAUSE.finditer(before))
        after_clause = re.match(r"\s*[,:(]?\s*khoan\s+(\d+[a-z]?)\b", after)
        clause = before_clauses[-1][1] if before_clauses else after_clause[1] if after_clause else None
        points = list(_POINT.finditer(before))
        following_points = list(_POINT.finditer(after)) if after_clause else []
        point = points[-1][1] if points else following_points[0][1] if following_points else None
        start, end = article.start(), article.end()
        raw = text[offsets[start]:offsets[end - 1] + 1]
        result.append(LegalReference(document, article[1], clause, point, raw,
                                     nearest is None and document_hint is not None))
    for mention in documents:
        if mention not in used_documents:
            start, end, key = mention
            result.append(LegalReference(key, raw=text[offsets[start]:offsets[end - 1] + 1]))
    if not result and document_hint:
        result.append(LegalReference(document_hint, document_inferred_from_metadata=True))
    unique = {}
    for ref in result:
        unique.setdefault((ref.document, ref.article, ref.clause, ref.point), ref)
    return list(unique.values())


def source_references(source: Mapping[str, Any]) -> list[LegalReference]:
    metadata = source.get("metadata") or {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    title = "\n".join(str(metadata.get(k) or "") for k in ("document_name", "retrieval_title"))
    mentions = _document_mentions(fold(title))
    keys = {m[2] for m in mentions}
    hint = next(iter(keys)) if len(keys) == 1 else None
    text = str(source.get("text") or source.get("passage") or "")
    return extract_references(text, document_hint=hint)


def compare_references(reference: list[LegalReference], observed: Iterable[LegalReference]) -> dict[str, Any]:
    observed = list(observed)
    scoped = [r for r in reference if r.document]
    documents = {r.document for r in scoped}
    seen_documents = {r.document for r in observed if r.document}
    covered = []
    for target in scoped:
        match = any(target.document == value.document
                    and (target.article is None or target.article == value.article)
                    and (target.clause is None or target.clause == value.clause)
                    and (target.point is None or target.point == value.point) for value in observed)
        covered.append(match)
    return {
        "reference_document_count": len(documents),
        "matched_document_count": len(documents & seen_documents),
        "reference_anchor_count": len(scoped), "matched_anchor_count": sum(covered),
        "document_coverage": len(documents & seen_documents) / len(documents) if documents else None,
        "anchor_coverage": sum(covered) / len(scoped) if scoped else None,
        "unresolved_reference_anchor_count": sum(r.document is None for r in reference),
        "observed_references": [asdict(r) for r in observed],
        "missing_reference_anchors": [asdict(r) for r, ok in zip(scoped, covered) if not ok],
        "interpretation": "weak_reference_anchor_coverage; not document Recall@K or proof of answer correctness",
    }


def compare_prediction_references(prediction: Mapping[str, Any]) -> dict[str, Any]:
    reference = extract_references(str(prediction["reference_answer"]))
    answer = prediction.get("prediction", {})
    context = prediction.get("context", {}).get("sources", [])
    cited_ids = {str(c.get("source_id")) for c in answer.get("citations", [])}
    stages = {
        "answer": extract_references(str(answer.get("answer", ""))),
        "reranked_top_chunks": [r for source in prediction.get("retrieved_chunks", []) for r in source_references(source)],
        "selected_context": [r for source in context for r in source_references(source)],
        "cited_context": [r for source in context if str(source.get("source_id")) in cited_ids for r in source_references(source)],
    }
    return {"extractor_version": EXTRACTOR_VERSION,
            "reference_references": [asdict(r) for r in reference],
            "answer_comparison_applicable": not bool(answer.get("abstained")),
            **{stage: compare_references(reference, refs) for stage, refs in stages.items()}}
