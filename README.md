# Vietnamese Legal IR Prototype

Production-oriented prototype extracted from the research repository. It searches a
Qdrant collection containing **long legal chunks** with the same BM25 and dense
retrieval semantics, deduplicates the lane union, attaches weighted-RRF
provenance, and reranks every retained chunk with the fine-tuned cross-encoder.

It intentionally contains no training, mining, or notebooks. Offline QA
evaluation exercises the same online service. Grounded QA is an inference-only layer over
retrieval. Offline production ingestion is isolated from online inference under
`legal_ir_prototype.ingestion`.

## Monorepo

The FastAPI backend lives at the repository root; the Next.js frontend lives in
[`frontend/`](frontend/). Both applications keep their own dependencies and
environment files.

```text
legal-rag-prototype/
├── src/legal_ir_prototype/    # Python backend
├── configs/
├── tests/
├── evals/                    # Frozen benchmark inputs
├── docker/
├── pyproject.toml
└── frontend/                 # Next.js frontend
    ├── src/
    ├── package.json
    └── package-lock.json
```

Run backend commands from the repository root. In a separate terminal, start
the frontend with:

```bash
cd frontend
npm ci
cp .env.example .env.local
npm run dev
```

Set `NEXT_PUBLIC_API_BASE_URL` in `frontend/.env.local` to the backend URL. The
backend's `LEGAL_RAG_CORS_ORIGINS` must include the frontend origin.

For Vercel, connect this repository and set **Root Directory** to `frontend`,
with build command `npm run build` and install command `npm ci`. Configure
`NEXT_PUBLIC_API_BASE_URL` in the Vercel project. The backend retains the root
Docker build context and can be deployed independently.

Backend checks run from the root with `python -m pytest`; frontend checks run
inside `frontend/` with `npm run lint` and `npm run build`.

`runs/`, generated judge results, `docs/`, local environments and model artifacts
are ignored by Git. Frozen benchmark inputs under `evals/` remain versioned.


## Evaluation on performance and observability

This fine-tuned model increase recall@5 from 90 to 95 (+5) on internal test and ~94.7 on private test in the competition.
The following experiments were run on Mac M4 MPS.



### Latency per case (seconds)

LLM response latency uses the successful HTTP attempt for each case, excluding
failed attempts and retry/pacing waits. It measures client-observed time to
receive the full response, including network and provider processing.

| Stage | Avg/case | P50 | P95 | Max |
|---|---:|---:|---:|---:|
| Qdrant dense | 0.242 | 0.210 | 0.516 | 1.493 |
| Qdrant sparse | 0.031 | 0.025 | 0.060 | 0.128 |
| Retrieval before reranking | 0.518 | 0.418 | 1.086 | 2.866 |
| Reranker | 25.384 | 22.622 | 43.199 | 52.612 |
| Retrieval total, including reranking | 25.905 | 22.987 | 44.191 | 53.557 |
| LLM response, excluding retry/wait | 4.403 | 3.401 | 9.357 | 41.732 |
| Full pipeline, including retry/wait | 30.810 | 27.244 | 52.036 | 134.324 |

Retrieval total includes reranking; these stages are not additive. Full pipeline
latency retains retry/wait time as recorded by the service.

### Token usage

Reported generation usage across the 300 cases. Pricing is left blank for manual
entry.

| Token usage | Total | Avg/case | Total pricing |
|---|---:|---:|---:|
| Input | 2,118,613 | 7,062.043 | |
| Output | 51,336 | 171.120 | |

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

The local research correspondence notes are kept in
`docs/research_inference_audit.md` (ignored by Git).

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

## Offline QA evaluation

For multiple samples **in one prompt**, evidence/behavior diagnosis, legal-anchor
comparison and explicit error rates, use the new `judge-diagnostic` command:

```bash
legal-rag-eval judge-diagnostic \
  --config configs/judge_large.yaml \
  --predictions runs/qa_eval/prototype300_run1/predictions.jsonl \
  --output runs/qa_eval/prototype300_run1/judge_diagnostic_v3.jsonl \
  --samples-per-prompt 5 --continue
```

It keeps v2 outputs intact, persists results per case, and writes an analysis
directory beside its output. No generation/retrieval is repeated. Details,
classification rules, limits of inferred legal references and the separate
offline `analyze` command are in the local, Git-ignored
`docs/diagnostic_evaluation.md`.

The current frozen benchmark contains **250 labeled QA questions plus 50
guardrail/abstention probes** in
`evals/datasets/prototype300_v1/cases.jsonl`. The original QA answers are retained;
behavior probes use explicit rubrics. See [`evals/README.md`](evals/README.md)
for selection, provenance, slices and limitations.

Use `--continue` on both the initial invocation and subsequent invocations:
it skips successful cases, retries previously failed cases, and processes the
remaining cases without restarting completed work.

```bash
legal-rag-eval run \
  --config configs/runtime.yaml \
  --env-file .env \
  --dataset evals/datasets/prototype300_v1/cases.jsonl \
  --output runs/qa_eval/prototype300_run1 \
  --continue

legal-rag-eval judge \
  --config configs/judge_large.yaml \
  --env-file .env \
  --predictions runs/qa_eval/prototype300_run1/predictions.jsonl \
  --output runs/qa_eval/prototype300_run1/judge_results_v2.jsonl \
  --continue
```

Generation and judging share bounded timeout/HTTP retries, exponential backoff
with jitter, HTTP `Retry-After` and Gemini `RetryInfo` support. Progress is
fsynced after each case; resume can recover a truncated final JSONL record,
preserving its bytes in a backup. Only an unfinished case may need another
provider call after an abrupt crash. `generation_trace.jsonl` and
`judge_results_v2.trace.jsonl` record each HTTP attempt, latency and reported
token usage. Optional explicit token prices enable cost estimates; absent
prices remain `null`. Full commands, recovery behavior and cost limitations
are documented in the local, Git-ignored `docs/evaluation.md`.

The earlier 30-case development workflow remains supported:

`legal-rag-eval` runs the exact online `RAGService` in one process, so the
dense encoder and reranker are loaded once. Evaluation questions, answers,
retrieved chunks, selected context and per-stage timings are written to
offline artifacts; they are deliberately not added to production request
logs.

Freeze the initial 30-case QA development set once:

```bash
legal-rag-eval prepare \
  --source /Users/w4ng/xquang/DSC_Legal/QA/train.json \
  --sample-size 30 \
  --seed 2026 \
  --output evals/datasets/qa_random30_v0
```

Run generation sequentially and resume safely after an interruption:

```bash
legal-rag-eval run \
  --config configs/runtime.yaml \
  --env-file .env \
  --dataset evals/datasets/qa_random30_v0/cases.jsonl \
  --output runs/qa_eval/qa_random30_run1 \
  --resume
```

Before spending 30 generation calls, validate the real dependencies and
artifact schema with a separate three-case smoke run:

```bash
legal-rag-eval run \
  --config configs/runtime.yaml \
  --env-file .env \
  --dataset evals/datasets/qa_random30_v0/cases.jsonl \
  --output runs/qa_eval/qa_random30_smoke3 \
  --limit 3
```

The default artifacts omit the exact prompt and provider raw response. Opt in
only for a controlled local evaluation with `--include-prompt` and/or
`--include-raw-llm-response`.

Run deterministic validation followed by one structured LLM judge call per
successful prediction:

```bash
legal-rag-eval judge \
  --predictions runs/qa_eval/qa_random30_run1/predictions.jsonl \
  --output runs/qa_eval/qa_random30_run1/judge_results_v2.jsonl \
  --resume
```

Transient provider failures are retained as append-only attempt history. Retry
only the currently failed generation cases without repeating successful calls:

```bash
legal-rag-eval run \
  --config configs/runtime.yaml \
  --env-file .env \
  --dataset evals/datasets/qa_random30_v0/cases.jsonl \
  --output runs/qa_eval/qa_random30_run1 \
  --resume \
  --retry-errors
```

Retry only failed judge calls in the same way:

```bash
legal-rag-eval judge \
  --config configs/judge_large.yaml \
  --env-file .env \
  --predictions runs/qa_eval/qa_random30_run1/predictions.jsonl \
  --output runs/qa_eval/qa_random30_run1/judge_results_large_v2.jsonl \
  --resume \
  --retry-errors
```

Summary metrics always use the newest attempt per `case_id`; a historical 503
stays in the JSONL audit trail but is no longer counted after a successful
retry.

New runs save an immutable resolved runtime snapshot. The judge uses that
snapshot unless `--config` is explicit; legacy config paths are hash-checked.
Judge v2 understands behavioral rubrics and must use a new output file when
upgrading a v1 run. `runs/` is ignored by Git, while the small frozen dataset
and its source hash can be versioned.


