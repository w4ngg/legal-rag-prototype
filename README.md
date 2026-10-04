# Vietnamese Legal IR Prototype

Production-oriented prototype extracted from the research repository. It searches a
Qdrant collection containing **long legal chunks** with the same BM25 and dense
retrieval semantics, deduplicates the lane union, attaches weighted-RRF
provenance, and reranks every retained chunk with the fine-tuned cross-encoder.

It intentionally contains no training, mining, notebooks, offline evaluation,
or experiment code. Grounded QA is implemented as an inference-only layer over
retrieval. Offline production ingestion is isolated from online inference under
`legal_ir_prototype.ingestion`.

## Package layout

```text
src/legal_ir_prototype/
├── core/           # typed config and device/dtype runtime helpers
├── domain/         # LegalChunk, RetrievalHit and SparseQueryVector
├── model/          # dense encoder, sparse encoder and reranker
├── vectorstore/    # Qdrant adapter and payload/schema validation
├── fusion/         # weighted-RRF candidate fusion
├── services/       # retrieval use-case orchestration
├── rag/            # context, prompts, citations, evidence gates, orchestration
├── llm/            # fake and OpenAI-compatible structured-output clients
├── api/            # FastAPI routes, runtime ownership and Prometheus metrics
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

After retrieval, `RAGService.answer(query)` builds a bounded citable context,
calls the configured structured-output LLM, resolves every citation ID against
the server-owned context map, and applies deterministic evidence gates. Unknown
or missing citations never reach the public answer. Empty context, failed
citation validation, or an LLM `insufficient_evidence` decision returns a safe
abstention instead.

Score-based evidence thresholds are optional and disabled until calibrated on a
validation set. Raw reranker logits and LLM-reported confidence are not treated
as probabilities.

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
cp .env.example .env
chmod 600 .env
```

Edit `configs/runtime.yaml`. Model IDs, revisions, Qdrant URL/collection/vector
names, artifact path, top-k values, weights, model lengths and device/dtype all
live in this file; none are embedded in implementation code.

Edit `.env` and replace the LLM key placeholder. The file is ignored by Git.
`LEGAL_RAG_CORS_ORIGINS` is a comma-separated allowlist of exact frontend
origins; wildcard origins are rejected. Values already present in the process
environment take precedence over `.env`.

## CLI smoke test

Retrieval only:

```bash
legal-ir-smoke \
  --config configs/runtime.yaml \
  --query 'Người khiếu nại có quyền và nghĩa vụ gì?' \
  --limit 5
```

`--limit` only limits displayed results. It does not change lane retrieval,
fusion, or reranking.

End-to-end grounded RAG (the YAML must include an `llm` section):

```bash
legal-rag-smoke \
  --config configs/runtime.yaml \
  --env-file .env \
  --query 'Người khiếu nại có quyền và nghĩa vụ gì?' \
  --show-context
```

With `llm.provider: fake`, the same command exercises the complete local
retrieval/context/citation path without making an external LLM request.
`--show-context` is an opt-in smoke/debug flag that adds selected and excluded
sources, the rendered context, token estimate, and exact system/user prompts to
the JSON output. Omit it when evidence text must not be printed or logged.

## FastAPI service

Start one API process (one process owns one dense model and one reranker):

```bash
legal-rag-api \
  --config configs/runtime.yaml \
  --env-file .env \
  --host 0.0.0.0 \
  --port 8000 \
  --require-cuda
```

On macOS, omit `--require-cuda`. For local browser testing, the example `.env`
allows `http://localhost:3000` and `http://127.0.0.1:3000`. Add the exact HTTPS
Vercel production origin before deployment; do not use `*`.

The HTTP contract is:

- `POST /v1/chat`: public grounded RAG answer with validated citations.
- `GET /health`: lightweight process liveness; never contacts a dependency.
- `GET /ready`: model, Qdrant, optional CUDA and LLM readiness; returns 503 when
  any required dependency is unavailable.
- `GET /metrics`: Prometheus exposition format.
- `GET /version`: deployed models, revisions, collection and schema identity.
- `POST /v1/retrieve`: raw internal retrieval diagnostics.

Raw retrieval evidence is disabled by default. Enable it only on a private
network and require an internal key:

```bash
export LEGAL_RAG_ENABLE_RETRIEVE=true
export LEGAL_RAG_INTERNAL_API_KEY='generate-a-long-random-secret'

curl -sS http://127.0.0.1:8000/v1/retrieve \
  -H 'Content-Type: application/json' \
  -H "X-Internal-API-Key: $LEGAL_RAG_INTERNAL_API_KEY" \
  -d '{"query":"sa thải trái pháp luật","top_k":5}'
```

For `/v1/chat`, the optional inbound `X-Request-ID` is echoed in the response;
otherwise the service creates one. `LEGAL_RAG_MAX_CONCURRENT_INFERENCE`
defaults to `1` to avoid concurrent GPU memory spikes. Run a single Uvicorn
worker per GPU because additional workers duplicate both models in VRAM.

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

## Vercel demo through Cloudflare Tunnel

For a temporary demo, the Vercel frontend can call a FastAPI process running
on the local Mac through a Cloudflare Quick Tunnel. First verify that FastAPI
is healthy locally:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/ready
```

Then keep a separate terminal open and create the tunnel. HTTP/2 is selected
explicitly because it is often more reliable than QUIC on restricted networks:

```bash
cloudflared tunnel \
  --protocol http2 \
  --url http://127.0.0.1:8000
```

Copy the generated `https://<random-name>.trycloudflare.com` URL and verify it:

```bash
export LEGAL_RAG_PUBLIC_API="https://<random-name>.trycloudflare.com"

curl -fsS "$LEGAL_RAG_PUBLIC_API/health"
curl -fsS "$LEGAL_RAG_PUBLIC_API/ready"
```

Set the following environment variable in the Vercel project for Production
and, when applicable, Preview:

```text
NEXT_PUBLIC_API_BASE_URL=https://<random-name>.trycloudflare.com
```

Redeploy the frontend after changing this variable. The backend CORS allowlist
must contain the exact frontend origin, not the tunnel URL:

```dotenv
LEGAL_RAG_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,https://<actual-project>.vercel.app
```

Restart FastAPI after changing `.env`.

### Troubleshooting `Unauthorized: Tunnel not found`

The following message normally means that an old ephemeral Quick Tunnel is no
longer registered with Cloudflare:

```text
ERR Register tunnel error from server side error="Unauthorized: Tunnel not found"
```

Do not leave that process retrying. Recover as follows:

1. Stop the old `cloudflared` process with `Ctrl+C`.
2. Confirm `/health` and `/ready` still work on `127.0.0.1:8000`.
3. Run the `cloudflared tunnel --protocol http2 --url ...` command above.
4. Test the newly generated public URL.
5. Replace `NEXT_PUBLIC_API_BASE_URL` in Vercel with the new URL.
6. Redeploy the frontend and test `/v1/chat` from the deployed UI.

A Quick Tunnel URL is temporary, changes when the tunnel is recreated, and
stops working when the `cloudflared` process or the Mac stops. Use a named
Cloudflare Tunnel with a stable hostname for a persistent demo. If the error
comes from a named tunnel instead, verify that the tunnel still exists in the
Cloudflare dashboard and run it with its current credentials/token; a deleted
tunnel or an old rotated token cannot register a new connection.
