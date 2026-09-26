# Qdrant ingestion and AWS migration

This package supports two intentionally separate paths:

1. **First migration:** snapshot the already validated local hybrid collection,
   copy the snapshot to S3, then restore it on the AWS Qdrant instance. This is
   the preferred first deployment because no embeddings or sparse weights are
   recomputed.
2. **Future rebuilds:** raw context JSON is converted to deterministic long
   chunks, BM25 document weights are computed from the complete corpus, dense
   passages are encoded in batches, and both named vectors are upserted directly
   to a new Qdrant collection. No FAISS index is created.

The `ingestion/` package is offline tooling. The online service never imports it.

## Data contract

Raw documents have the existing shape:

```json
{"id": "156", "name": "...", "link": "...", "passage": "..."}
```

`metadata.legal` is optional. Verified fields such as chapters, sections,
articles, clauses, points and references are retained when present; otherwise
the Qdrant `legal` object is deliberately empty. The ingester never invents
legal structure from unverified text.

Qdrant points contain the configured named `dense` and `sparse` vectors plus
payload schema version 1. Point IDs are UUIDv5 values derived from `chunk_id`, so
replaying an acknowledged batch replaces the same points rather than creating
duplicates.

## Install

```bash
python -m pip install -e '.[ingestion]'
```

Edit `configs/runtime.yaml` for either local Qdrant or the private AWS endpoint.
All model IDs/revisions, named vectors, collection name and device settings stay
in runtime YAML. Edit `configs/ingestion.yaml` only for chunking, BM25, Qdrant
HNSW build parameters and upload behavior.

## Preferred first migration: local snapshot to AWS

Create and download a snapshot from local Qdrant:

```bash
legal-ir-ingest snapshot-create \
  --config configs/ingestion.yaml \
  --output-dir artifacts/snapshots \
  --collection-name legal_long_hybrid_v1
```

The command writes the `.snapshot` file, copies the matching sparse query bundle
into `sparse_query/`, and writes a manifest containing all checksums, Qdrant
version, source collection schema and point count. Qdrant snapshots do not
contain these client-side query artifacts, so deploy both the snapshot and the
copied bundle.

Upload the immutable snapshot to S3 and create a short-lived URL:

```bash
aws s3 cp artifacts/snapshots/<file>.snapshot \
  s3://<bucket>/qdrant/<file>.snapshot
aws s3 presign s3://<bucket>/qdrant/<file>.snapshot --expires-in 3600
```

From a host configured for the AWS Qdrant endpoint, restore into a **new**
collection:

```bash
legal-ir-ingest snapshot-restore \
  --config configs/ingestion.aws.yaml \
  --snapshot-url '<presigned-s3-url>' \
  --target-collection legal_long_hybrid_v1
```

The Qdrant EC2 instance itself downloads the URL, so its security group, route
and S3 access path must permit that request. Use matching Qdrant versions for
snapshot portability. Switch application traffic only after a smoke test; keep
the previous collection until rollback is no longer required.

## Future rebuild: raw data directly to Qdrant

Run preparation independently when desired:

```bash
legal-ir-ingest prepare \
  --config configs/ingestion.yaml \
  --raw-dir /data/selected-contexts \
  --work-dir artifacts/ingestion-v1
```

Then encode and upload:

```bash
legal-ir-ingest ingest \
  --config configs/ingestion.yaml \
  --chunks artifacts/ingestion-v1/chunks/long_chunks.jsonl \
  --state-dir artifacts/ingestion-v1/qdrant/legal_long_hybrid_v2 \
  --collection-name legal_long_hybrid_v2
```

Or run both steps:

```bash
legal-ir-ingest run \
  --config configs/ingestion.yaml \
  --raw-dir /data/selected-contexts \
  --work-dir artifacts/ingestion-v2 \
  --collection-name legal_long_hybrid_v2
```

The upload can be resumed with the exact same command and work directory.
`--overwrite-preparation` rebuilds chunks/BM25 state and should only be used
intentionally; it does not delete or overwrite an existing Qdrant collection.

After a successful run, point `sparse.query_encoder_manifest` in the serving
runtime config to:

```text
<state-dir>/sparse_query/manifest.json
```

The final ingestion manifest records chunk hash/count, model pin, vector names,
payload schema, optimizer status and three sampled point validations.
