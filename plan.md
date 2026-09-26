# Kế hoạch triển khai Legal RAG production-oriented

## 1. Mục tiêu

Xây dựng hệ thống Legal RAG có thể triển khai và vận hành trên AWS, với các đặc tính chính:

- Retrieval hybrid trên Qdrant: dense retrieval, sparse retrieval, fusion và reranking.
- Embedding model và reranker chạy bằng CUDA trên một GPU EC2.
- Qdrant chạy độc lập trên một CPU EC2, lưu dữ liệu bền vững trên EBS.
- Model được tải từ Hugging Face Hub và luôn pin bằng commit SHA; không phụ thuộc vào trạng thái hiện tại của branch `main`.
- LLM được gọi qua giao thức OpenAI-compatible và có thể đổi DeepSeek, Gemini hoặc provider khác bằng cấu hình.
- Mọi release liên kết rõ ràng giữa source code, container image, model, Qdrant index, prompt và bộ evaluation.
- Có kiểm thử, quan sát hệ thống, CI/CD, acceptance gate, promote và rollback.

Tài liệu này ưu tiên một production baseline thực dụng. Những hạng mục như Kubernetes, Qdrant cluster nhiều node, autoscaling phức tạp và multi-region disaster recovery chưa nằm trong phạm vi V1.

## 2. Nguyên tắc thiết kế

1. **Tách compute theo workload**: model inference dùng GPU; vector database dùng CPU, RAM và disk I/O.
2. **Không có version ngầm định**: model, image, index, prompt và dataset đều có version bất biến.
3. **Model-index compatibility**: một Qdrant index chỉ được dùng với đúng embedding model và preprocessing đã tạo ra nó.
4. **Candidate trước, production sau**: build và đánh giá trên candidate collection; chỉ chuyển alias sau khi qua gate.
5. **Rollback không rebuild**: rollback bằng image digest, release manifest và Qdrant alias đã tồn tại.
6. **Private by default**: GPU EC2 và Qdrant EC2 nằm trong private subnet; Qdrant không public ra Internet.
7. **Provider-agnostic ở tầng LLM**: business logic không import SDK riêng của DeepSeek/Gemini.
8. **Fail closed**: service không nhận traffic nếu model, index hoặc release manifest không tương thích.
9. **Không log dữ liệu pháp lý nhạy cảm mặc định**: log metadata kỹ thuật, hash và ID; không log raw query/context/answer nếu chưa có chính sách dữ liệu.

## 3. Kiến trúc tổng thể

```text
                            Internet / Client
                                   |
                            Route 53 + TLS
                                   |
                           ALB / API Gateway
                                   |
                         private application subnet
                                   |
                    +--------------v---------------+
                    | GPU EC2: Legal RAG service   |
                    |------------------------------|
                    | FastAPI                      |
                    | query preprocessing          |
                    | embedding model -> CUDA      |
                    | sparse query encoder         |
                    | retrieval orchestration      |
                    | fusion                       |
                    | reranker -> CUDA             |
                    | context builder              |
                    | LLM adapter                  |
                    | citation/abstention guards   |
                    +---------+-----------+--------+
                              |           |
           private VPC traffic|           | HTTPS egress
                              |           |
              +---------------v--+     +--v----------------------+
              | CPU EC2: Qdrant  |     | OpenAI-compatible LLM  |
              |------------------|     | DeepSeek / Gemini / ... |
              | dense vectors    |     +-------------------------+
              | sparse vectors   |
              | payload indexes  |
              | EBS data volume  |
              +------------------+

   Control and artifact plane:

   GitHub Actions -> ECR -> deployment by SSM
   Hugging Face Hub -> pinned model snapshot -> GPU EC2 model cache
   S3 -> release manifests, eval datasets/results, MLflow artifacts, backups
   Prometheus/Grafana/CloudWatch/MLflow <- metrics, logs and evaluation runs
```

### 3.1 GPU EC2

Trách nhiệm:

- Chạy FastAPI và toàn bộ retrieval pipeline.
- Load embedding model và reranker lên GPU.
- Tạo dense query vector và sparse query representation.
- Gọi Qdrant qua private IP/DNS.
- Thực hiện fusion, reranking, context building, LLM generation, citation validation và abstention.
- Export application metrics và GPU metrics.

Khởi điểm đề xuất: instance thuộc họ `g5` hoặc loại tương đương, chọn size sau benchmark. Không chốt `g5.xlarge` trước khi đo VRAM, CPU, RAM, concurrency và latency thực tế.

Storage:

- Root volume chỉ chứa hệ điều hành và container runtime.
- Model cache nằm trên EBS riêng hoặc instance storage nếu chấp nhận tải lại khi replace instance.
- Model cache không phải source of truth; Hugging Face revision và release manifest mới là source of truth.

### 3.2 CPU EC2 cho Qdrant

Trách nhiệm:

- Chạy duy nhất Qdrant và các exporter/agent liên quan.
- Lưu collection, WAL và snapshot trên EBS.
- Chỉ nhận kết nối từ security group của GPU EC2 và đường quản trị được cho phép.

Storage:

- Mount EBS vào đường dẫn ổn định, ví dụ `/var/lib/qdrant`.
- Dùng volume có hiệu năng IOPS/throughput được cấu hình rõ ràng; theo dõi queue depth, latency và free space.
- Bật EBS snapshot theo lịch và kiểm thử restore định kỳ.
- Không coi EBS snapshot là cơ chế versioning duy nhất của index; index version vẫn được quản lý bằng collection bất biến và release manifest.

### 3.3 Observability và MLflow

V1 có thể dùng stack giám sát dùng chung của tổ chức hoặc một control-plane host riêng. Không nên đặt Prometheus, Grafana và MLflow workload nặng chung với Qdrant production vì có thể tranh CPU, RAM và disk I/O.

Thiết kế mục tiêu:

- Prometheus scrape GPU service, Qdrant và DCGM exporter.
- Grafana đọc Prometheus.
- CloudWatch nhận system logs, application logs và EC2 alarms.
- MLflow Tracking Server lưu metadata trong PostgreSQL/RDS và artifacts trong S3.
- Nếu V1 chưa có RDS, có thể bắt đầu bằng một MLflow deployment nhỏ nhưng phải lên kế hoạch backup và migration; không dùng local SQLite cho production multi-user.

## 4. Luồng xử lý online

```text
Request
  -> validate/auth/rate limit
  -> normalize query
  -> dense embedding + sparse encoding
  -> Qdrant dense/sparse search
  -> fusion
  -> optional metadata/ACL filtering
  -> rerank candidates
  -> context deduplication and token budgeting
  -> OpenAI-compatible LLM call
  -> parse structured output
  -> citation validation
  -> confidence/abstention policy
  -> response + trace/metrics
```

Quy tắc quan trọng:

- Không đưa document vào context nếu caller không có quyền truy cập payload đó.
- Citation chỉ hợp lệ nếu trỏ tới chunk thực sự được đưa vào context.
- Nếu LLM trả JSON không hợp lệ, thực hiện tối đa số lần repair đã cấu hình rồi fail có kiểm soát.
- Nếu retrieval score/evidence coverage dưới ngưỡng, trả lời abstain thay vì ép LLM suy đoán.
- Retry LLM chỉ áp dụng với lỗi transient và phải có exponential backoff, jitter, timeout cùng retry budget.

## 5. Cấu trúc repository

```text
legal-rag/
├── README.md
├── plan.md
├── pyproject.toml
├── uv.lock                         # hoặc lockfile tương đương
├── Makefile
├── .env.example
├── .gitignore
├── configs/
│   ├── base.yaml
│   ├── local.yaml
│   ├── staging.yaml
│   ├── production.yaml
│   ├── prompts/
│   │   ├── answer_v1.yaml
│   │   └── citation_repair_v1.yaml
│   └── releases/
│       └── example.release.yaml
├── src/legal_rag/
│   ├── __init__.py
│   ├── app.py                     # FastAPI application factory
│   ├── settings.py                # typed config and env loading
│   ├── lifecycle.py               # startup/readiness/shutdown
│   ├── api/
│   │   ├── dependencies.py
│   │   ├── errors.py
│   │   ├── middleware.py
│   │   ├── schemas.py
│   │   └── routes/
│   │       ├── answer.py
│   │       ├── search.py
│   │       ├── health.py
│   │       └── metadata.py
│   ├── domain/
│   │   ├── models.py
│   │   ├── citations.py
│   │   └── exceptions.py
│   ├── retrieval/
│   │   ├── pipeline.py
│   │   ├── dense.py
│   │   ├── sparse.py
│   │   ├── fusion.py
│   │   ├── reranker.py
│   │   ├── filters.py
│   │   └── thresholds.py
│   ├── ingestion/
│   │   ├── loaders.py
│   │   ├── normalize.py
│   │   ├── chunking.py
│   │   ├── identifiers.py
│   │   ├── embed.py
│   │   ├── upsert.py
│   │   ├── validate.py
│   │   └── checkpoints.py
│   ├── generation/
│   │   ├── client.py              # OpenAI-compatible abstraction
│   │   ├── prompts.py
│   │   ├── schemas.py
│   │   ├── context_builder.py
│   │   ├── citation_validator.py
│   │   └── abstention.py
│   ├── models/
│   │   ├── hub.py                 # pinned HF download/cache
│   │   ├── embedding.py
│   │   ├── reranker.py
│   │   └── compatibility.py
│   ├── qdrant/
│   │   ├── client.py
│   │   ├── schema.py
│   │   ├── collections.py
│   │   ├── aliases.py
│   │   └── snapshots.py
│   ├── observability/
│   │   ├── logging.py
│   │   ├── metrics.py
│   │   └── tracing.py
│   └── security/
│       ├── auth.py
│       └── redaction.py
├── scripts/
│   ├── download_models.py
│   ├── create_collection.py
│   ├── ingest_corpus.py
│   ├── validate_collection.py
│   ├── run_ir_eval.py
│   ├── run_rag_eval.py
│   ├── build_candidate.py
│   ├── promote_release.py
│   ├── rollback_release.py
│   └── smoke_test.py
├── evals/
│   ├── datasets/
│   │   ├── ir/
│   │   └── rag/
│   ├── metrics/
│   ├── judges/
│   ├── thresholds/
│   └── reports/
├── manifests/
│   ├── models/
│   ├── indexes/
│   └── releases/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── contract/
│   ├── regression/
│   ├── load/
│   └── fixtures/
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.ingestion
│   │   └── compose.local.yaml
│   ├── qdrant/
│   │   ├── compose.yaml
│   │   └── config.yaml
│   ├── systemd/
│   ├── cloud-init/
│   ├── terraform/                 # VPC, EC2, EBS, IAM, ALB, ECR, S3
│   ├── prometheus/
│   └── grafana/
├── .github/workflows/
│   ├── ci.yaml
│   ├── build-image.yaml
│   ├── deploy-staging.yaml
│   ├── deploy-production.yaml
│   ├── build-candidate.yaml
│   └── rollback.yaml
└── docs/
    ├── architecture.md
    ├── runbooks/
    │   ├── deploy.md
    │   ├── rollback.md
    │   ├── qdrant-restore.md
    │   ├── llm-provider-outage.md
    │   └── gpu-oom.md
    └── adr/
```

### 5.1 Ranh giới module

- `retrieval/` không biết HTTP và không gọi LLM.
- `generation/` nhận danh sách evidence đã chuẩn hóa; không truy cập Qdrant trực tiếp.
- `qdrant/` chịu trách nhiệm schema, client và alias; không chứa business policy.
- `models/` chịu trách nhiệm tải, kiểm tra và load artifact; không quyết định acceptance threshold.
- `ingestion/` dùng chung normalization/chunking contract với evaluation nhưng chạy như job độc lập, không chạy trong web process.
- `api/` chỉ điều phối, validate và chuyển lỗi domain thành HTTP response.

## 6. Quản lý cấu hình và secrets

### 6.1 Cấu hình YAML không chứa secret

Ví dụ `configs/production.yaml`:

```yaml
service:
  environment: production
  log_level: INFO
  request_timeout_seconds: 45

release:
  manifest_uri: s3://legal-rag-releases/production/current.yaml

retrieval:
  dense_top_k: 50
  sparse_top_k: 50
  fusion_top_k: 40
  rerank_top_k: 12
  final_context_k: 8
  fusion_method: rrf
  rrf_k: 60

context:
  max_input_tokens: 12000
  reserved_output_tokens: 1500
  deduplicate_by: parent_section_id

llm:
  protocol: openai_compatible
  base_url: https://provider.example/v1
  model: provider-model-name
  timeout_seconds: 30
  max_retries: 2
  temperature: 0.0
  response_format: json_schema

qdrant:
  url: http://qdrant.internal:6333
  collection_alias: legal_chunks_current
  request_timeout_seconds: 5
```

Config trên chỉ là ví dụ. Giá trị top-k, token budget, timeout và threshold phải được chốt bằng benchmark/evaluation.

### 6.2 Environment variables

| Biến | Bắt buộc | Ý nghĩa |
|---|---:|---|
| `LEGAL_RAG_ENV` | Có | `local`, `staging`, `production` |
| `LEGAL_RAG_CONFIG_PATH` | Có | Đường dẫn file cấu hình |
| `RELEASE_MANIFEST_URI` | Production | URI manifest đang active; có thể override YAML khi deploy |
| `HF_TOKEN` | Với private repo | Token read-only lấy từ Secrets Manager |
| `HF_HOME` | Có | Model cache directory |
| `QDRANT_URL` | Có | Private URL của Qdrant |
| `QDRANT_API_KEY` | Khuyến nghị | API key, lấy từ Secrets Manager |
| `OPENAI_BASE_URL` | Có | Endpoint OpenAI-compatible của provider |
| `OPENAI_API_KEY` | Có | Provider key, lấy từ Secrets Manager |
| `OPENAI_MODEL` | Có | Tên model tại provider |
| `AWS_REGION` | Có | Region triển khai |
| `MLFLOW_TRACKING_URI` | Offline jobs | MLflow server URI |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Nếu dùng tracing | Trace collector endpoint |

Quy tắc:

- `.env` chỉ dùng local và không commit.
- Secret được lưu trong AWS Secrets Manager hoặc SSM Parameter Store dạng `SecureString`.
- EC2 nhận secret bằng instance role có quyền tối thiểu, không dùng access key tĩnh.
- Không đưa token vào container image, user data, GitHub secret output hoặc log.
- Config được validate bằng typed settings ngay khi process khởi động; thiếu hoặc sai kiểu thì process dừng.

### 6.3 Đổi LLM provider

Adapter chỉ phụ thuộc các trường OpenAI-compatible cơ bản:

- `base_url`
- `api_key`
- `model`
- chat/completions hoặc responses-like endpoint đã chọn cho V1
- timeout, retry và generation parameters

Mỗi provider cần một contract test vì mức độ tương thích có thể khác nhau ở JSON schema, tool calling, streaming, token usage và error format. Provider-specific differences được cô lập trong `generation/client.py`, không lan sang pipeline.

## 7. Hugging Face model artifact management

### 7.1 Tổ chức model

Dùng hai Hugging Face repository riêng:

```text
organization/legal-embedding
organization/legal-reranker
```

Production manifest luôn dùng full commit SHA, không dùng `main`, branch động hoặc tag có thể bị di chuyển:

```yaml
embedding:
  repo_id: organization/legal-embedding
  revision: 84d7130000000000000000000000000000000000
  artifact_sha256: "..."
  dimension: 1024
  normalize: true

reranker:
  repo_id: organization/legal-reranker
  revision: 91ab420000000000000000000000000000000000
  artifact_sha256: "..."
  max_length: 512
```

Tag như `v1.2.0` hữu ích cho con người, nhưng release manifest vẫn ghi commit SHA được resolve từ tag lúc candidate build.

### 7.2 Luồng tải model lên GPU EC2

1. Process đọc release manifest từ S3.
2. Validate schema và chữ ký/checksum của manifest nếu có.
3. Dùng `snapshot_download` với `repo_id` và full `revision`.
4. Tải vào staging cache directory, không ghi đè snapshot đang active.
5. Kiểm tra file bắt buộc, checksum, model config, embedding dimension và tokenizer.
6. Chỉ cho phép `trust_remote_code` khi code đã được audit và commit SHA đã pin; mặc định tắt.
7. Load model từ local snapshot với chế độ offline.
8. Chuyển model lên CUDA, chạy warmup và smoke inference.
9. Kiểm tra model/index compatibility với Qdrant active collection.
10. Chỉ đánh dấu readiness sau khi tất cả bước thành công.

Nên prefetch model trong bước deploy trước khi switch traffic để startup không phụ thuộc hoàn toàn vào tốc độ tải từ Hub. Giữ lại ít nhất snapshot hiện tại và snapshot trước đó để rollback nhanh; cleanup theo dung lượng và release retention policy.

## 8. Qdrant schema và ingestion

### 8.1 Collection strategy

Không ghi trực tiếp vào collection production. Mỗi index là một concrete collection bất biến:

```text
legal_chunks__idx_2026_09_25_001
legal_chunks__idx_2026_10_02_001

alias:
legal_chunks_current -> legal_chunks__idx_2026_09_25_001
```

Candidate build tạo collection mới. Promote là thao tác đổi alias có kiểm soát sau acceptance gate.

### 8.2 Vector schema

Một point đại diện cho một legal chunk:

- Named dense vector, ví dụ `dense`, dimension lấy từ embedding manifest.
- Named sparse vector, ví dụ `sparse`, nếu hybrid retrieval được bật.
- Distance metric của dense vector được khóa trong index manifest; với normalized embeddings thường dùng cosine nhưng phải khớp baseline.
- HNSW, quantization, on-disk vector và optimizer settings phải được benchmark trước khi production.

Payload tối thiểu:

```text
point_id               deterministic UUID/hash
document_id
document_version
chunk_id
parent_section_id
source_uri
jurisdiction
document_type
effective_from
effective_to
language
title
section_path
text
text_sha256
acl_tags
ingested_at
corpus_snapshot_id
chunker_version
```

Tạo payload index cho các trường thực sự dùng để filter, ví dụ `jurisdiction`, `document_type`, `effective_from`, `effective_to`, `language`, `acl_tags` và `document_id`. Không tạo index payload tràn lan.

### 8.3 ID và idempotency

- `document_id` ổn định theo nguồn nghiệp vụ.
- `document_version` thay đổi khi nội dung nguồn thay đổi.
- `chunk_id` được tạo xác định từ `document_id + document_version + chunker_version + section_path + chunk_ordinal`.
- `point_id` được suy ra deterministically từ `chunk_id`.
- Rerun cùng corpus snapshot và cùng pipeline phải tạo cùng point set.
- Ingestion job ghi checkpoint và có thể resume theo batch.

### 8.4 Ingestion flow

```text
source snapshot
  -> validate input
  -> normalize text/metadata
  -> legal-aware chunking
  -> deterministic IDs
  -> dense embedding in batches
  -> sparse encoding
  -> create candidate collection
  -> batched upsert with retry/checkpoint
  -> payload/vector count validation
  -> sampled retrieval validation
  -> snapshot candidate collection
  -> write index manifest
```

Các validation bắt buộc:

- Số document/chunk mong đợi và số point thực tế.
- Không có vector sai dimension, NaN hoặc rỗng.
- Tỷ lệ metadata thiếu dưới ngưỡng.
- Text checksum và duplicate rate.
- Filter theo jurisdiction/effective date/ACL hoạt động đúng.
- Search smoke test trả đúng point cho tập known-item queries.

## 9. Release manifest và versioning

### 9.1 Release manifest

Một release là đơn vị promote/rollback, không phải riêng container image:

```yaml
schema_version: 1
release_id: rag_2026_09_25_001
created_at: "2026-09-25T08:00:00Z"

code:
  git_commit: abcdef1234567890
  image_uri: 123456789012.dkr.ecr.region.amazonaws.com/legal-rag@sha256:...

embedding:
  repo_id: organization/legal-embedding
  revision: 84d7130000000000000000000000000000000000
  artifact_sha256: "..."
  dimension: 1024
  normalize: true

sparse_encoder:
  name: sparse-model-or-algorithm
  version: "..."

reranker:
  repo_id: organization/legal-reranker
  revision: 91ab420000000000000000000000000000000000
  artifact_sha256: "..."

index:
  index_version: idx_2026_09_25_001
  collection: legal_chunks__idx_2026_09_25_001
  expected_alias: legal_chunks_current
  corpus_snapshot_id: corpus_2026_09_24
  chunker_version: legal_chunker_v3
  embedding_revision: 84d7130000000000000000000000000000000000
  qdrant_schema_hash: "..."

generation:
  prompt_version: answer_v1
  output_schema_version: answer_schema_v1
  llm_protocol: openai_compatible
  llm_model: provider-model-name

evaluation:
  ir_dataset_version: ir_gold_v4
  rag_dataset_version: rag_gold_v3
  thresholds_version: acceptance_v2
  mlflow_run_ids:
    - run-id-ir
    - run-id-rag
```

Manifest được validate bằng JSON Schema/Pydantic và lưu ở vị trí bất biến trên S3. Con trỏ `production/current.yaml` chỉ trỏ tới/copy từ manifest đã được chấp nhận; lịch sử release không bị ghi đè.

### 9.2 Compatibility checks

Startup phải từ chối readiness nếu có một trong các lỗi:

- Embedding revision trong release khác revision trong index manifest.
- Embedding dimension khác Qdrant vector size.
- Sparse encoder hoặc preprocessing version không khớp.
- Active Qdrant alias không trỏ tới collection được release mong đợi.
- Output schema/prompt version không được image hiện tại hỗ trợ.
- Artifact checksum không đúng.

## 10. API contract

Tất cả endpoint nghiệp vụ dùng prefix `/v1`. API response có `request_id`, release metadata tối thiểu và error schema thống nhất.

### 10.1 `POST /v1/search`

Dùng cho debug retrieval có kiểm soát hoặc consumer chỉ cần evidence.

Request:

```json
{
  "query": "Điều kiện đơn phương chấm dứt hợp đồng là gì?",
  "filters": {
    "jurisdiction": ["VN"],
    "document_type": ["law"],
    "effective_at": "2026-09-25"
  },
  "top_k": 10
}
```

Response rút gọn:

```json
{
  "request_id": "req_...",
  "results": [
    {
      "chunk_id": "...",
      "document_id": "...",
      "title": "...",
      "section_path": "...",
      "text": "...",
      "score": 0.91,
      "source_uri": "..."
    }
  ],
  "release_id": "rag_2026_09_25_001"
}
```

Trong production, endpoint này cần authorization riêng vì có thể trả raw legal text và retrieval scores.

### 10.2 `POST /v1/answer`

Request:

```json
{
  "query": "Điều kiện đơn phương chấm dứt hợp đồng là gì?",
  "filters": {
    "jurisdiction": ["VN"],
    "effective_at": "2026-09-25"
  },
  "conversation": [],
  "options": {
    "stream": false
  }
}
```

Response:

```json
{
  "request_id": "req_...",
  "answer": "...",
  "citations": [
    {
      "citation_id": "C1",
      "chunk_id": "...",
      "document_id": "...",
      "title": "...",
      "section_path": "...",
      "source_uri": "..."
    }
  ],
  "abstained": false,
  "abstention_reason": null,
  "release_id": "rag_2026_09_25_001"
}
```

Không public raw confidence như xác suất đúng nếu score chưa được calibrate. Có thể trả `evidence_strength` dạng category sau khi đã đánh giá calibration.

### 10.3 Health và metadata

| Endpoint | Mục đích | Kiểm tra dependency |
|---|---|---|
| `GET /health/live` | Process còn sống | Không gọi dependency ngoài |
| `GET /health/ready` | Có thể nhận traffic | Model loaded, CUDA usable, Qdrant reachable, release compatible |
| `GET /metrics` | Prometheus scrape | Bảo vệ bằng private network/security group |
| `GET /v1/metadata` | Release/model/index đang chạy | Chỉ trả metadata không nhạy cảm |

Readiness phải fail khi mất Qdrant kéo dài, model chưa load hoặc compatibility check thất bại. Liveness không nên fail chỉ vì provider LLM tạm thời lỗi, tránh restart loop.

### 10.4 Error schema

```json
{
  "request_id": "req_...",
  "error": {
    "code": "UPSTREAM_TIMEOUT",
    "message": "Không thể hoàn thành yêu cầu trong thời gian cho phép",
    "retryable": true
  }
}
```

Các nhóm lỗi tối thiểu: validation, authentication/authorization, rate limit, retrieval unavailable, provider timeout, invalid provider response, citation validation failure, capacity/GPU OOM và internal error. Không trả stack trace hoặc secret cho client.

## 11. Context builder, structured output và safety guards

### 11.1 Context builder

- Reranked candidates được deduplicate theo document/section.
- Giữ hierarchy và source metadata để tạo citation ổn định.
- Áp dụng token budget trước khi gọi LLM; không cắt giữa câu/điều luật nếu có thể.
- Ưu tiên evidence score cao nhưng giữ diversity giữa các nguồn khi policy yêu cầu.
- Tách system instruction, user query và evidence bằng cấu trúc rõ ràng.
- Coi document text là untrusted data; prompt phải yêu cầu bỏ qua instruction nằm trong evidence.

### 11.2 Structured output

Schema nội bộ nên gồm:

- `answer`
- `citations[]` với citation ID/chunk ID
- `abstained`
- `abstention_reason`
- `claims[]` nếu cần kiểm tra citation theo từng claim

Validate bằng Pydantic/JSON Schema. Nếu provider không hỗ trợ JSON schema đáng tin cậy, dùng JSON mode hoặc parse/repair có giới hạn và ghi metric riêng.

### 11.3 Citation validation

- Mọi citation ID phải tồn tại trong context map.
- Không chấp nhận chunk/document ID do LLM tự tạo.
- Với answer không abstain, các factual legal claims phải có citation theo policy.
- Source metadata trong response lấy từ server-side context map, không lấy nguyên từ output LLM.
- Citation failure vượt quá repair budget phải trả lỗi có kiểm soát hoặc abstain, không trả answer chưa kiểm chứng.

### 11.4 Abstention

Abstention policy kết hợp:

- Không có candidate sau metadata/ACL filter.
- Top reranker score hoặc score margin dưới threshold đã calibrate.
- Evidence coverage không đủ cho câu hỏi nhiều phần.
- Citation validation thất bại.
- Tài liệu không còn hiệu lực tại thời điểm người dùng hỏi nếu không có nguồn phù hợp.
- Provider response không hợp lệ sau retry/repair.

Threshold phải nằm trong versioned config và được chọn trên validation set; không hard-code rải rác trong source.

## 12. Startup, shutdown và deploy flow

### 12.1 Application startup

1. Load typed config và secrets.
2. Fetch và validate release manifest.
3. Resolve/download pinned Hugging Face snapshots.
4. Verify checksums và compatibility metadata.
5. Load tokenizer, embedding model và reranker.
6. Move model lên CUDA; kiểm tra VRAM headroom.
7. Kết nối Qdrant và đọc alias/schema/index manifest.
8. Kiểm tra model-index compatibility.
9. Chạy warmup embedding/reranking và known-query smoke test.
10. Khởi tạo LLM client; chạy lightweight contract check nếu policy/cost cho phép.
11. Export release info metric.
12. Chuyển readiness sang ready.

Startup có timeout rõ ràng. Nếu lỗi, process dừng với structured log thay vì chạy ở trạng thái một phần.

### 12.2 Graceful shutdown

- Readiness chuyển fail trước.
- ALB ngừng gửi request mới.
- Chờ in-flight requests trong giới hạn grace period.
- Đóng HTTP/Qdrant clients.
- Giải phóng model resources và flush telemetry.

### 12.3 Deployment

1. CI build một image và push ECR bằng immutable digest.
2. Candidate release đã có manifest và evaluation evidence.
3. Deploy image + manifest URI lên staging.
4. Prefetch model, warmup và smoke/load test.
5. Sau approval, deploy production theo rolling hoặc blue/green.
6. Instance mới chỉ vào target group khi readiness pass.
7. Chạy post-deploy synthetic queries và kiểm tra dashboards.
8. Giữ deployment trước và model cache trước trong rollback window.

Không dùng mutable image tag như `latest` làm định danh deploy. Tag có thể dùng cho tiện ích, nhưng deployment phải ghi digest.

### 12.4 Rollback online

Rollback release phải khôi phục đồng bộ:

1. Chọn release manifest trước đó đã pass gate.
2. Đảm bảo concrete Qdrant collection của release đó còn tồn tại.
3. Chuyển Qdrant alias về collection cũ nếu alias đã đổi.
4. Deploy image digest và model revisions của release cũ.
5. Chạy compatibility check, readiness và smoke test.
6. Ghi audit event và lý do rollback.

Thứ tự chi tiết giữa alias switch và application switch phụ thuộc chiến lược blue/green. An toàn nhất là mỗi deployment dùng collection concrete từ manifest trong thời gian chuyển đổi; chỉ đổi shared alias khi tất cả consumer tương thích.

## 13. Security và networking

### 13.1 Network layout

- ALB ở public subnets, chỉ mở `443` ra Internet; redirect hoặc đóng `80`.
- GPU EC2 ở private application subnet, không có public IP.
- Qdrant EC2 ở private data subnet, không có public IP.
- Qdrant port `6333` chỉ cho phép inbound từ security group của GPU EC2 và monitoring scraper cần thiết.
- Port quản trị không mở Internet; dùng AWS Systems Manager Session Manager.
- Outbound từ GPU EC2 chỉ cho phép các destination cần thiết: Hugging Face, LLM provider, AWS services hoặc đi qua controlled egress/NAT/proxy.
- Cân nhắc VPC endpoints cho ECR, S3, CloudWatch, Secrets Manager và SSM để giảm public egress.

### 13.2 IAM

- GitHub Actions dùng OIDC federation, không lưu AWS access key dài hạn.
- GPU EC2 role chỉ được pull ECR, đọc đúng secret, đọc release manifest/model-related S3 objects và ghi logs/metrics cần thiết.
- Qdrant EC2 role chỉ có quyền snapshot/backup cần thiết.
- Candidate/evaluation job role tách khỏi online serving role.
- Promote role tách khỏi build role và được bảo vệ bằng environment approval.

### 13.3 Application security

- Xác thực bằng JWT/API key tùy consumer; authorization được kiểm tra trước retrieval.
- Rate limit ở gateway/ALB companion layer hoặc middleware phù hợp.
- Request body, query length, filter cardinality và `top_k` có giới hạn.
- ACL filter được áp dụng server-side; client không được tự khai báo quyền.
- TLS cho external traffic; cân nhắc TLS/API key cho Qdrant ngay cả trong VPC nếu threat model yêu cầu.
- Pin dependencies, scan image/dependencies và chạy container bằng non-root user.
- Không mount Docker socket vào service container.
- Mã hóa EBS và S3 bằng KMS; bật rotation/retention phù hợp.
- Redact authorization header, API key, token, raw query và document text khỏi log.
- Có chính sách retention, deletion và audit cho dữ liệu pháp lý/người dùng.

## 14. Observability

### 14.1 Structured logs

Log JSON với các trường:

```text
timestamp, level, service, environment, request_id, trace_id,
release_id, route, status_code, latency_ms, error_code,
llm_provider, llm_model, qdrant_collection, retry_count
```

Không dùng query, document ID có tính nhạy cảm hoặc error message từ provider làm Prometheus label. Nếu cần điều tra nội dung, dùng cơ chế audit riêng có quyền truy cập và retention rõ ràng.

### 14.2 Metrics chính

API:

- `legal_rag_http_requests_total{route,status_class}`
- `legal_rag_http_request_duration_seconds{route}`
- `legal_rag_inflight_requests`
- `legal_rag_errors_total{component,error_code}`

Retrieval:

- `legal_rag_embedding_duration_seconds`
- `legal_rag_qdrant_search_duration_seconds{search_type}`
- `legal_rag_retrieval_candidates_count{stage}`
- `legal_rag_rerank_duration_seconds`
- `legal_rag_rerank_batch_size`
- `legal_rag_empty_retrieval_total`
- `legal_rag_abstentions_total{reason}`

Generation:

- `legal_rag_llm_requests_total{provider,model,outcome}`
- `legal_rag_llm_duration_seconds{provider,model}`
- `legal_rag_llm_input_tokens_total{provider,model}`
- `legal_rag_llm_output_tokens_total{provider,model}`
- `legal_rag_llm_retries_total{provider,reason}`
- `legal_rag_structured_output_failures_total{provider}`
- `legal_rag_citation_validation_failures_total{reason}`

GPU/host:

- GPU utilization, memory used/free, temperature, power, throttling và XID errors qua DCGM exporter.
- CPU, RAM, disk utilization, disk latency, EBS queue depth và network.
- Process restart count và model load/warmup time.

Qdrant:

- Search latency/error rate.
- Collection point/vector count.
- Memory, CPU, disk usage và free space.
- WAL/optimizer/indexing state.
- Snapshot age và snapshot failure count.

Release info:

- Một info metric chứa `release_id`, short model revisions, `index_version`, `prompt_version` và image digest rút gọn.
- Không dùng release ID có cardinality không kiểm soát trên dashboard lịch sử dài; giới hạn retention/label strategy.

### 14.3 SLO và alert baseline

Chốt SLO sau load test, nhưng tối thiểu cần alert cho:

- Availability/readiness giảm.
- p95/p99 latency vượt budget liên tục.
- Error rate hoặc LLM timeout tăng.
- Citation failure/abstention rate thay đổi bất thường.
- GPU OOM, GPU unavailable hoặc VRAM gần đầy.
- Qdrant disk gần đầy, search errors hoặc snapshot quá cũ.
- EBS latency/queue depth cao.
- Không có metrics/logs từ instance.

Alert phải trỏ tới runbook và phân biệt warning/critical. Không alert theo từng request lỗi đơn lẻ.

### 14.4 MLflow

Mỗi offline run log:

- Git commit và container image digest.
- Release/model/index/prompt/dataset versions.
- Toàn bộ retrieval/generation parameters.
- IR metrics theo overall và slice.
- RAG metrics theo overall và slice.
- Latency, throughput, VRAM/RAM usage và estimated provider cost.
- Evaluation report, error analysis samples đã redact và acceptance decision.

Không dùng MLflow Model Registry thay thế release manifest; MLflow lưu evidence/thí nghiệm, còn manifest là deployment contract.

## 15. Test strategy

### 15.1 Unit tests

- Normalization, chunking và deterministic IDs.
- Fusion/ranking với input cố định.
- Context token budgeting và deduplication.
- Citation mapping/validation.
- Abstention policy và threshold boundaries.
- Config/release manifest validation.
- Provider error mapping, retry và timeout.
- Model-index compatibility checks.

Các test này không cần GPU, network hoặc provider thật.

### 15.2 Integration tests

- Khởi động Qdrant test container và tạo collection thật.
- Ingest fixture corpus, dense/sparse search, filter và alias switch.
- FastAPI gọi Qdrant qua test client.
- Model nhỏ/test double cho embedding/reranker ở CI thường.
- GPU integration suite chạy trên self-hosted GPU runner hoặc scheduled environment.

### 15.3 Contract tests

- Hugging Face snapshot có đủ file và load được tại pinned revision.
- Qdrant schema/API tương thích với client version.
- Mỗi LLM provider được cấu hình trả đúng structured output và usage/error fields kỳ vọng.
- API OpenAPI schema backward-compatible hoặc có version bump.

Không gọi provider thật trên mọi unit-test PR. Dùng mock server trong PR và chạy provider contract test có quota theo lịch/pre-release.

### 15.4 Regression tests

- Golden IR queries với expected relevant document/chunk IDs.
- Known-answer/citation cases.
- No-answer và out-of-domain cases.
- Effective-date, jurisdiction, ACL và multilingual slices.
- Prompt injection nằm trong document text.
- Provider malformed JSON, timeout, rate limit và partial failure.

### 15.5 Offline evaluation

IR metrics:

- Recall@K là gate chính trước reranking.
- MRR@K, nDCG@K, Precision@K và reranker uplift.
- Zero-result rate, latency và metrics theo slice.

RAG metrics:

- Citation precision/recall hoặc citation validity/coverage.
- Faithfulness/groundedness.
- Answer correctness/relevance trên human-labeled set.
- Abstention precision/recall.
- Structured-output success rate.
- Safety/prompt-injection behavior.
- End-to-end latency và estimated cost per answer.

LLM-as-judge không phải ground truth duy nhất. Pin judge model/prompt/version, lưu raw judge output, calibrate bằng human review và theo dõi variance.

### 15.6 Load, soak và failure tests

- Đo single-request latency trước, sau đó tăng concurrency tới target.
- Tách latency budget: embedding, Qdrant, rerank, context, LLM và overhead.
- Kiểm tra micro-batching nếu có, VRAM headroom và GPU OOM recovery.
- Soak test để phát hiện memory leak/cache growth.
- Mô phỏng Qdrant unavailable, LLM timeout/rate limit, Hugging Face unavailable khi startup và disk gần đầy.
- Xác nhận timeout/circuit breaker/backpressure, không để queue tăng vô hạn.

## 16. CI/CD flow

### 16.1 Pull request CI

```text
checkout
  -> dependency lock verification
  -> formatting/lint/type check
  -> unit tests + coverage threshold
  -> integration tests with Qdrant
  -> OpenAPI/schema compatibility check
  -> secret/dependency/container/IaC scans
  -> build container
  -> smoke start with mocked dependencies
```

PR không được tải model production nhiều GB hoặc gọi LLM tính phí. Dùng fixture model/mock server; full GPU/provider tests chạy trong pre-release pipeline.

### 16.2 Build image

- Trigger khi merge vào protected branch hoặc tạo release.
- Build một lần, scan một lần, promote cùng image digest qua staging và production.
- Gắn metadata: git SHA, build time, source repo và SBOM.
- Push ECR immutable tag theo git SHA và lưu digest.
- Ký image nếu tổ chức hỗ trợ image signing.

### 16.3 Deploy staging

- GitHub Actions assume AWS role qua OIDC.
- Cập nhật deployment bằng SSM/Auto Scaling launch template hoặc cơ chế IaC đã chọn.
- Truyền image digest và candidate release manifest URI.
- Chờ readiness, chạy smoke tests, synthetic answer/citation tests và load sanity test.
- Upload kết quả về MLflow/S3.

### 16.4 Deploy production

- Chỉ nhận release đã pass acceptance gate.
- Bảo vệ bằng GitHub Environment approval.
- Blue/green là mục tiêu; rolling deployment chấp nhận cho V1 nếu capacity cho phép và rollback đã kiểm thử.
- Sau deploy, theo dõi error, latency, GPU, abstention và citation metrics trong observation window.
- Tự động rollback cho lỗi kỹ thuật rõ ràng; chất lượng semantic cần human decision trừ khi gate xác định chắc chắn.

### 16.5 Qdrant deployment

Qdrant config/image cũng phải pin version. Upgrade Qdrant tách khỏi content index promote:

1. Đọc compatibility/migration notes.
2. Snapshot và xác minh restore path.
3. Test bằng bản sao staging của collection.
4. Upgrade trong maintenance window hoặc theo HA strategy tương lai.
5. Validate point count, search, filters và latency sau upgrade.

## 17. Offline MLOps pipeline

### 17.1 Candidate build

Input bất biến:

- Corpus snapshot ID.
- Embedding/reranker Hugging Face commit SHA.
- Sparse encoder version.
- Chunker/preprocessing version.
- Qdrant schema config.
- Source git commit và ingestion image digest.

Pipeline:

```text
validate inputs
  -> resolve and pin artifacts
  -> create candidate collection
  -> ingest and validate
  -> write index manifest
  -> run IR evaluation
  -> run RAG evaluation
  -> run performance/cost checks
  -> evaluate acceptance policy
  -> emit candidate release manifest and signed report
```

Candidate job không tự promote production chỉ vì pipeline chạy thành công.

### 17.2 Acceptance gate

Gate được version hóa, ví dụ `acceptance_v2.yaml`, và có ba nhóm:

1. **Hard correctness gates**: schema, point count, no NaN, citation validity, ACL/effective-date tests phải pass tuyệt đối.
2. **Quality gates**: Recall@K, nDCG, answer correctness, faithfulness và abstention đạt absolute threshold, đồng thời không regression quá tolerance so với champion.
3. **Operational gates**: p95 latency, throughput, GPU memory, Qdrant resource usage, error rate và cost không vượt budget.

Đánh giá overall và các critical slices. Không cho average tốt che regression lớn ở một jurisdiction/document type quan trọng.

### 17.3 Promote

1. Xác nhận candidate manifest, MLflow runs và approval.
2. Giữ snapshot của Qdrant candidate.
3. Publish release manifest bất biến.
4. Deploy candidate vào staging và chạy post-deploy checks.
5. Chuyển production bằng blue/green hoặc quy trình đã chuẩn hóa.
6. Chuyển alias Qdrant atomically tại điểm thích hợp.
7. Cập nhật production pointer.
8. Ghi audit record: ai, lúc nào, release nào, evidence nào.

### 17.4 Rollback và retention

- Giữ tối thiểu release hiện tại và một số release đã biết tốt trước đó.
- Không xóa concrete collection/model cache ngay sau promote.
- Retention policy dựa trên dung lượng, compliance và rollback window.
- Rollback drill định kỳ phải chứng minh thời gian khôi phục, không chỉ tồn tại runbook.
- Nếu corpus có yêu cầu xóa dữ liệu bắt buộc, retention/rollback phải tuân thủ deletion policy thay vì giữ vô hạn.

## 18. Các wave triển khai và Definition of Done

### Wave 1 — IR baseline, refactor và Qdrant ingestion

#### Công việc

1. Đóng băng research baseline: code commit, dataset snapshot, qrels, config, random seed, model revisions và metric report.
2. Chuyển notebook/research code thành module có interface rõ ràng cho embedding, sparse encoding, fusion và reranking.
3. Viết unit tests bảo toàn output/ranking quan trọng của baseline.
4. Định nghĩa Qdrant named-vector schema, payload schema, filter indexes và collection naming.
5. Xây legal-aware normalization/chunking cùng deterministic IDs.
6. Xây ingestion job có batching, retry, checkpoint, resume và idempotency.
7. Tạo candidate collection, ingest fixture rồi full corpus snapshot.
8. Viết validation cho counts, dimension, checksum, duplicates, filters và known-item search.
9. Chạy regression IR giữa baseline research và Qdrant pipeline; giải thích mọi chênh lệch.
10. Ghi index manifest đầu tiên.

#### Definition of Done

- [ ] Baseline có thể chạy lại từ commit/config/dataset đã pin.
- [ ] Research notebook không còn là dependency của production pipeline.
- [ ] Ingestion rerun không tạo point trùng và có thể resume sau lỗi.
- [ ] Qdrant schema/payload/filter indexes được test tự động.
- [ ] Dense, sparse, fusion và reranking đều có regression tests.
- [ ] Recall@K/nDCG/MRR đạt ngưỡng đã ghi nhận hoặc regression được phê duyệt.
- [ ] Candidate collection có index manifest và validation report.
- [ ] Alias switch và rollback alias đã được test trên staging/local.

### Wave 2 — AWS GPU inference và Hugging Face artifacts

#### Công việc

1. Dùng Terraform tạo VPC/subnets/security groups/IAM/ECR/S3/Secrets Manager và hai EC2 roles.
2. Provision GPU EC2 với NVIDIA driver/container toolkit hoặc DLAMI đã pin và được kiểm thử.
3. Provision CPU EC2 cho Qdrant, attach/mount encrypted EBS và cấu hình backup.
4. Đưa Qdrant vào private data subnet; chỉ cho GPU security group truy cập.
5. Tạo private Hugging Face repos nếu cần và push embedding/reranker artifact chuẩn.
6. Pin commit SHA, checksum và model metadata trong manifest.
7. Xây downloader/cache/warmup flow; mặc định local/offline load sau download.
8. Benchmark embedding và reranker theo batch size, sequence length, concurrency, precision và VRAM.
9. Benchmark Qdrant latency/throughput và EBS behavior với dataset gần production.
10. Chọn instance size và serving parameters dựa trên kết quả đo.

#### Definition of Done

- [ ] GPU EC2 và Qdrant EC2 nằm trên private subnet, không có public Qdrant port.
- [ ] EC2 được quản trị bằng SSM, không phụ thuộc SSH public.
- [ ] Model được tải từ đúng full commit SHA và checksum được xác minh.
- [ ] Restart service không đổi model ngoài ý muốn.
- [ ] Embedding/reranker chạy CUDA, warmup pass và có VRAM headroom đã định nghĩa.
- [ ] Qdrant data tồn tại sau container/instance service restart.
- [ ] EBS snapshot và restore thử nghiệm thành công.
- [ ] Benchmark report xác định capacity, bottleneck và cấu hình ban đầu.

### Wave 3 — RAG V1

#### Công việc

1. Xây retrieval orchestrator: dense/sparse search, fusion, filter và rerank.
2. Xây context builder có deduplication, section hierarchy và token budget.
3. Xây OpenAI-compatible client và provider config cho ít nhất một provider chính, một provider thay thế.
4. Version prompt và structured output schema.
5. Xây citation map, validator và repair policy.
6. Xây abstention policy dựa trên evidence và failure state.
7. Thêm timeout, retry, backoff, rate-limit handling và provider error normalization.
8. Tạo RAG golden set gồm answerable, unanswerable, time-sensitive, ACL và adversarial cases.
9. Chạy offline RAG evaluation và human review sample.

#### Definition of Done

- [ ] Đổi provider bằng config, không đổi retrieval/business code.
- [ ] Provider contract tests pass cho các provider được hỗ trợ.
- [ ] Structured output parse thành công đạt ngưỡng đã định nghĩa.
- [ ] Citation không thể trỏ ngoài context và metadata citation do server cấp.
- [ ] No-answer/out-of-domain cases abstain theo ngưỡng chấp nhận.
- [ ] Prompt injection trong evidence có regression test.
- [ ] IR và RAG metrics được log vào MLflow với đầy đủ version metadata.
- [ ] Có baseline latency/cost/quality cho từng provider đã hỗ trợ.

### Wave 4 — FastAPI serving, Docker và ECR

#### Công việc

1. Tạo FastAPI application factory, typed schemas và middleware request ID/auth/logging.
2. Implement `/v1/search`, `/v1/answer`, health, readiness, metadata và metrics endpoints.
3. Chuẩn hóa error taxonomy và response mapping.
4. Implement startup compatibility checks, model warmup và graceful shutdown.
5. Tạo production Dockerfile chạy non-root, pin dependency và có health behavior đúng.
6. Tạo local compose cho API + Qdrant + mock LLM; production vẫn chạy hai EC2 tách biệt.
7. Push image lên ECR bằng immutable digest và tạo SBOM/scan report.
8. Chạy smoke/load/failure tests.

#### Definition of Done

- [ ] OpenAPI schema được version/control và contract test.
- [ ] Liveness/readiness có semantics đúng; ALB chỉ route tới instance ready.
- [ ] Không có secret/model weights baked trong image.
- [ ] Container chạy non-root và image scan không còn critical finding chưa xử lý/phê duyệt.
- [ ] Timeout, request size, query length, `top_k` và concurrency có giới hạn.
- [ ] Graceful shutdown không làm rơi request trong grace period đã đặt.
- [ ] Image có thể được xác định chính xác bằng ECR digest và git SHA.

### Wave 5 — Observability

#### Công việc

1. Chuẩn hóa JSON logs, request/trace IDs và redaction.
2. Instrument API, retrieval, reranker, LLM, citation và abstention metrics.
3. Deploy node/system collectors, Qdrant metrics và NVIDIA DCGM exporter.
4. Cấu hình Prometheus scrape và Grafana dashboards.
5. Gửi logs/system metrics cần thiết sang CloudWatch.
6. Deploy/configure MLflow tracking với S3 artifacts và durable metadata store.
7. Định nghĩa SLO/alerts và liên kết runbooks.
8. Kiểm tra metric cardinality, log PII/secrets và observability cost.

#### Definition of Done

- [ ] Có dashboard end-to-end latency và latency theo từng stage.
- [ ] Có dashboard GPU/VRAM, Qdrant/EBS và LLM provider.
- [ ] Có metrics citation failure, abstention, structured-output failure và empty retrieval.
- [ ] Mỗi request có thể correlate qua logs/traces mà không cần log raw content.
- [ ] Alerts critical đã được fire/test và trỏ tới runbook.
- [ ] MLflow run liên kết được với git, image, model, index, prompt và datasets.
- [ ] Không có secret/raw sensitive content trong log/metric labels qua security review.

### Wave 6 — GitHub Actions CI/CD

#### Công việc

1. Thiết lập protected branches, required checks và CODEOWNERS phù hợp.
2. Xây PR CI: lint, type, unit, integration, contract, security và image smoke.
3. Xây ECR build workflow dùng OIDC, immutable digest, scan và SBOM.
4. Xây staging deploy workflow bằng SSM/IaC, readiness và synthetic tests.
5. Xây production workflow có environment approval và observation window.
6. Xây rollback workflow nhận `release_id`, không nhận tập version rời rạc.
7. Bảo vệ concurrency để hai production deploy không chạy chồng.
8. Kiểm thử credential scope và audit trail.

#### Definition of Done

- [ ] GitHub không lưu AWS long-lived access keys.
- [ ] PR không merge nếu quality/security checks bắt buộc fail.
- [ ] Một image digest duy nhất được promote từ staging sang production.
- [ ] Production deployment có approval, audit record và post-deploy verification.
- [ ] Rollback workflow đã chạy thử thành công với release thật trên staging.
- [ ] Concurrent production deploy bị serialize/cancel theo policy.
- [ ] CI không phụ thuộc vào live paid provider cho test thông thường.

### Wave 7 — Offline MLOps, promote và rollback

#### Công việc

1. Chuẩn hóa model, index, dataset, prompt, threshold và release manifests.
2. Xây candidate build workflow nhận toàn bộ input version bất biến.
3. Tự động build/validate Qdrant candidate collection.
4. Chạy IR evaluation theo overall và critical slices.
5. Chạy RAG evaluation, human review sample và operational benchmark.
6. Implement versioned acceptance policy với hard/quality/operational gates.
7. Tạo promote workflow chuyển alias/deployment/pointer có audit.
8. Tạo rollback workflow và retention policy.
9. Thực hiện rollback drill và Qdrant restore drill.

#### Definition of Done

- [ ] Có thể truy từ `release_id` tới mọi artifact/config/eval evidence.
- [ ] Candidate không thể promote nếu hard gate fail.
- [ ] Quality được so cả absolute threshold, champion delta và critical slices.
- [ ] Promote không rebuild model/index/image.
- [ ] Model revision và index embedding revision luôn được compatibility check.
- [ ] Production pointer, Qdrant alias và deployed service phản ánh cùng release.
- [ ] Rollback hoàn tất trong RTO mục tiêu đã đo.
- [ ] Collection/model/release retention và deletion policy được tài liệu hóa.

## 19. Thứ tự triển khai khuyến nghị

Các wave là mốc quản lý, nhưng một số nền tảng cần làm sớm để tránh refactor lớn:

1. **Freeze baseline và viết manifests schema trước**: xác định cái gì cần version ngay từ đầu.
2. **Refactor retrieval và build Qdrant ingestion local**: chưa cần GPU cloud để chứng minh correctness.
3. **Thiết lập AWS network/IAM/ECR/S3/Secrets nền tảng**.
4. **Đưa Qdrant lên CPU EC2 và xác minh EBS backup/restore**.
5. **Đưa pinned models lên GPU EC2, benchmark và chốt capacity**.
6. **Hoàn thiện RAG V1 với provider adapter, citation và abstention**.
7. **Đóng gói FastAPI/Docker, health/readiness và error handling**.
8. **Instrument metrics/logs ngay trước load test** để benchmark có dữ liệu phân rã.
9. **Hoàn thiện PR CI và image build**, sau đó staging deploy.
10. **Xây offline candidate/gate/promote/rollback pipeline** trên nền manifests đã dùng từ đầu.
11. **Chạy full staging rehearsal**: candidate build -> eval -> promote -> deploy -> rollback -> restore.
12. **Go-live có observation window**, capacity buffer và người trực xử lý sự cố.

## 20. Quyết định cần chốt bằng benchmark hoặc nghiệp vụ

Các giá trị sau không nên đoán trong code:

- Corpus size, update frequency và retention/deletion requirements.
- Embedding dimension, precision, batch size và normalization.
- Sparse encoder và fusion strategy/parameters.
- Dense/sparse top-k, rerank top-k và context top-k.
- Legal-aware chunk size/overlap và hierarchy rules.
- Instance types, EBS type/size/IOPS/throughput.
- Target QPS/concurrency, p95/p99 latency, availability SLO và cost budget.
- Acceptance thresholds theo overall và critical slices.
- LLM context/output token budget và supported provider capability matrix.
- Authentication, tenant isolation, ACL source of truth và audit retention.
- RPO/RTO, backup frequency và rollback retention window.

Mỗi quyết định quan trọng nên có ADR ngắn trong `docs/adr/`, gồm bối cảnh, lựa chọn, bằng chứng benchmark, hệ quả và điều kiện xem xét lại.

## 21. Go-live checklist tổng hợp

- [ ] Hai EC2 tách biệt: GPU service/model và CPU Qdrant/EBS.
- [ ] Không có Qdrant public endpoint; security groups và IAM đã review.
- [ ] Model Hugging Face được pin full commit SHA, checksum và load offline từ verified snapshot.
- [ ] Active release manifest liên kết image, models, index, prompt, datasets và eval runs.
- [ ] Model-index compatibility check chạy lúc startup và trước promote.
- [ ] Qdrant EBS snapshot, restore và alias rollback đã diễn tập.
- [ ] API auth, limits, timeouts, retry/backoff và graceful shutdown đã test.
- [ ] Citation validation, abstention và ACL/effective-date filters đã test trên golden sets.
- [ ] IR/RAG/operational gates pass theo critical slices.
- [ ] Dashboards, alerts và runbooks sẵn sàng; log không lộ secret/raw sensitive content.
- [ ] CI/CD dùng OIDC, image digest bất biến và production approval.
- [ ] Staging full rehearsal và production rollback drill hoàn tất.
- [ ] On-call owner, incident path, RPO/RTO và cost alarms đã được xác nhận.
