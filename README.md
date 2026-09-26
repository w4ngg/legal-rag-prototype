# Vietnamese Legal IR Prototype

Production-oriented prototype extracted from the research repository. It searches a
Qdrant collection containing **long legal chunks** with the same BM25 and dense
retrieval semantics, deduplicates the lane union, attaches weighted-RRF
provenance, and reranks every retained chunk with the fine-tuned cross-encoder.

It intentionally contains no training, mining, notebooks, offline evaluation,
QA generation, or experiment code. Offline production ingestion is isolated
from online inference under `legal_ir_prototype.ingestion`.

## Package layout

```text
src/legal_ir_prototype/
├── core/           # typed config and device/dtype runtime helpers
├── domain/         # LegalChunk, RetrievalHit and SparseQueryVector
├── model/          # dense encoder, sparse encoder and reranker
├── vectorstore/    # Qdrant adapter and payload/schema validation
├── fusion/         # weighted-RRF candidate fusion
├── services/       # retrieval use-case orchestration
├── ingestion/      # raw -> long chunks -> direct hybrid Qdrant + snapshots
└── cli.py          # command-line smoke entry point
```

The package root re-exports the public API. Compatibility modules such as
`legal_ir_prototype.dense` and `legal_ir_prototype.vector_store` remain
available so existing consumers do not break during the refactor.

## Runtime path

1. `DenseEncoder` strips the query and calls `encode_query` on
   `AITeamVN/Vietnamese_Embedding_v2`. Its commit, max length, normalization and
   runtime device/dtype are provided by YAML.
2. `SparseEncoder` uses the exact `bm25s` tokenizer and vocabulary exported by
   the Qdrant ingestion job. Query weights are term frequencies. Stored sparse
   document weights already contain BM25 Lucene weighting, so Qdrant IDF is not
   enabled.
3. `QdrantVectorStore` runs independent named-vector searches: dense DOT and
   sparse DOT. `exact: true` matches the research FAISS flat search semantics.
4. `FusionStrategy` canonicalizes each lane by descending raw score and
   `chunk_id`, keeps the best duplicate within a lane, then reproduces the two
   research fusion levels. First, lane-local MaxP ranks documents and weighted
   RRF selects candidate documents. Second, weighted chunk RRF selects evidence
   inside retained documents:

   ```text
   document_fusion(d) = sparse_weight / (rrf_k + sparse_doc_rank(d))
                      + dense_weight  / (rrf_k + dense_doc_rank(d))

   evidence_fusion(c) = sparse_weight / (rrf_k + sparse_chunk_rank(c))
                      + dense_weight  / (rrf_k + dense_chunk_rank(c))
   ```

   Missing-lane terms are zero. Raw BM25 and cosine/IP scores are **not** mixed
   or min-max/z-score normalized. Rank conversion is the score normalization.
   `candidate_documents` and `evidence_chunks_per_document` reproduce the
   research caps; either accepts `null` to remove that cap.
5. `Reranker` loads `w4ngg/VN_reranker_legal` at a pinned commit. This is the
   fine-tuned cross-encoder checkpoint whose base model was
   `AITeamVN/Vietnamese_Reranker`. It scores `(query, retrieval_text)` pairs and
   returns its raw one-label logit.
6. `RetrievalService.retrieve(query)` orders chunks by raw reranker logit, then
   uses fusion rank and `chunk_id` only to break exact ties. It returns ranked
   `LegalChunk` values with raw lane scores/ranks, fusion score/rank, reranker
   score/rank, and the complete Qdrant payload metadata.

There is deliberately **no document MaxP output**, sigmoid confidence
conversion, or fusion of a MaxP reranker score with a MaxP retrieval score.

The detailed correspondence to the research source is in
[`docs/research_inference_audit.md`](docs/research_inference_audit.md).

## Prerequisites

- Python 3.10+
- A Qdrant collection produced by the research repository's hybrid ingestion
  command. It must contain both configured named vectors and the long-chunk
  payload schema.
- The exported sparse query bundle (`manifest.json`, `vocab.index.json`, and
  `params.index.json`) from the same ingestion.
- Access to the two Hugging Face model repositories on the first run.

## Install and configure

```bash
cd legal-ir-prototype
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp configs/local.yaml configs/runtime.yaml
```

Edit `configs/runtime.yaml`. Model IDs, revisions, Qdrant URL/collection/vector
names, artifact path, top-k values, weights, model lengths and device/dtype all
live in this file; none are embedded in implementation code.

## CLI smoke test

```bash
legal-ir-smoke \
  --config configs/runtime.yaml \
  --query 'Người khiếu nại có quyền và nghĩa vụ gì?' \
  --limit 5
```

`--limit` only limits displayed results. It does not change lane retrieval,
fusion, or reranking.

## Tests

```bash
python -m pytest
```

The normal suite uses fakes and compact regression cases derived from three
representative research queries. It performs no network/model downloads.

To compare live Qdrant lane outputs with the preserved research
`deep_diagnostics.json`:

```bash
export LEGAL_IR_PROTOTYPE_CONFIG="$PWD/configs/runtime.yaml"
export LEGAL_IR_RESEARCH_DEEP_DIAGNOSTICS="/path/to/research/deep_diagnostics.json"
python -m pytest -m integration tests/test_live_research_parity.py
```

The live parity test compares dense@K and BM25-sparse@K chunk IDs for query IDs
`83864`, `151910`, and `154226`. Because it performs exact lane comparison, a
different corpus/index/model revision correctly fails the test.

## Ingestion and AWS migration

For the first deployment, migrate the already validated local Qdrant collection
by snapshot. For later corpus rebuilds, the new pipeline encodes dense vectors
and computes BM25 sparse weights directly into Qdrant without creating FAISS.
Commands, artifact contracts, resumability and S3/AWS restore steps are in
[`docs/ingestion.md`](docs/ingestion.md).
