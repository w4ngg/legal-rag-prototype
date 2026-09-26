# Research-to-prototype inference audit

## Scope audited

The online path was reconstructed from these research components:

- `retrieval/src/legal_ir/dense.py`
- `retrieval/src/legal_ir/bm25.py`
- `retrieval/src/legal_ir/fusion.py`
- `retrieval/src/legal_ir/reranker.py`
- `retrieval/src/legal_ir/pipeline.py`
- `retrieval/src/legal_ir/schema.py`
- `retrieval/src/legal_ir/io.py`
- `retrieval/src/legal_ir/ingest_qdrant.py`
- `runs/evaluate_long_only_bm25_40_dense_80/config/runtime_long_only_reranker.yaml`

The deployed prototype target is the long-only experiment: BM25@40 and
dense@80 search the same long-chunk corpus. HyDE and short-to-long mapping are
not in this serving path.

## Exact correspondence

| Concern | Research behavior retained in prototype |
|---|---|
| Query normalization | Reject empty input after `query.strip()`; otherwise preserve query text. |
| Dense preprocessing | `SentenceTransformer.encode_query`, `max_seq_length=2048`, normalized embedding, NumPy float32. No manually added prompt. |
| Dense model | `AITeamVN/Vietnamese_Embedding_v2` at `18b44161e041bf1d3a333ab5144b5b7b93f914d2`. |
| Dense similarity | Inner product over normalized embeddings. Qdrant uses named vector `dense`, DOT, and exact search by default. |
| Sparse preprocessing | `bm25s.tokenize`, lowercase, regex `(?u)\b\w\w+\b`, no stopwords, no stemmer, OOV dropped. Duplicate tokens become query term frequency. |
| Sparse search | Dot product against precomputed BM25S Lucene document weights. Qdrant's additional IDF modifier remains disabled. |
| Sparse artifacts | Vocabulary and BM25 params are exported beside the ingestion manifest and checksummed. The manifest vector name must match runtime Qdrant configuration. |
| Lane top-k | Dense 80 and BM25/sparse 40 in the supplied example YAML. Both are external configuration. |
| Lane ranking | Per-lane duplicate `chunk_id` keeps its highest finite score; sort by descending score then ascending `chunk_id`; ranks are one-based. |
| Candidate identity | `chunk_id`; payload conflicts for the same ID are rejected. The lane union is grouped by `document_id` only for research-compatible selection. |
| Fusion/normalization | Lane-local chunk→document MaxP, weighted document RRF for candidate documents, then weighted chunk RRF for evidence. `k=60`, weights 1.0/1.0. Ranks normalize incomparable raw scores. |
| Candidate cutoff | Example runtime keeps 120 candidate documents and 120 evidence chunks per document (effectively the full BM25@40+dense@80 union in the audited run). Either cap may be `null`. |
| Reranker architecture | Sentence-Transformers `CrossEncoder`, one-label raw output, Identity activation. `w4ngg/VN_reranker_legal` is the fine-tuned checkpoint; its base was `AITeamVN/Vietnamese_Reranker`. |
| Reranker input | Original stripped query paired with each chunk's `retrieval_text` (falling back to passage only at the domain-object boundary). |
| Reranker length/batch | 2304 tokens and batch size 4 in example YAML. |
| Reranker ranking | Descending raw logit; exact ties use pre-reranker fusion rank then `chunk_id`. No sigmoid calibration. |
| Returned result | Top 20 `LegalChunk` objects in example YAML, retaining retrieval/fusion/reranker scores and metadata. |

## Qdrant contract

Both named vectors are configuration values. The supplied development config
uses `dense` and `sparse`.

Required payload fields:

- `schema_version`, `chunk_id`, `document_id`, `vector_row`, `granularity`
- `passage`, `retrieval_text`
- `passage_sha256`, `embedding_text_sha256`

Optional payload fields retained in `LegalChunk.metadata` include document
name/title, chunk/character positions, source fields, chunking/normalization
versions, and the nested `legal` object (chapters, sections, articles, clauses,
points and verified references). The Qdrant point ID is also exposed as
`qdrant_point_id` metadata.

The service validates the dense dimension, DOT distance, both named vectors,
and the sparse bundle's checksum/scoring/tokenizer contract before use.

## Device and dtype

Both models accept external `device` and `dtype` configuration. `auto` selects
CUDA, then Apple MPS, then CPU. Auto dtype selects BF16 on capable CUDA, FP16 on
other accelerators, and FP32 on CPU. Hugging Face revisions are mandatory, so a
moving repository head cannot silently change inference.

## Intentional output boundary

The research competition pipeline eventually collapsed chunks to documents.
This online prototype's requested boundary is a ranked list of `LegalChunk`, so
it stops after chunk reranking. It does not run final document MaxP and does not
apply the experimental formula that combines a MaxP reranker rank/score with a
MaxP retrieval rank/score. Retrieval candidates, lane top-k values, query
encoders, per-lane ranking, deduplication and weighted-RRF provenance remain
unchanged.
