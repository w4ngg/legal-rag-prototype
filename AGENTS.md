Cấu trúc thư mục nên tổ chức hướng tới như này, với những file đã tạo không cần cố refractor
legal-ir-prototype/
│
├── src/
│   └── legal_ir/
│       │
│       ├── api/
│       │   ├── routes/
│       │   │   ├── chat.py
│       │   │   ├── health.py
│       │   │   └── metrics.py
│       │   │
│       │   ├── schemas.py
│       │   ├── dependencies.py
│       │   └── app.py
│       │
│       ├── retrieval/
│       │   ├── service.py
│       │   ├── types.py
│       │   ├── fusion.py
│       │   └── filters.py
│       │
│       ├── models/
│       │   ├── dense_encoder.py
│       │   ├── sparse_encoder.py
│       │   ├── reranker.py
│       │   └── device.py
│       │
│       ├── vectorstore/
│       │   ├── base.py
│       │   └── qdrant.py
│       │
│       ├── rag/
│       │   ├── service.py
│       │   ├── context_builder.py
│       │   ├── prompt_builder.py
│       │   ├── citation.py
│       │   └── abstention.py
│       │
│       ├── llm/
│       │   ├── base.py
│       │   └── openai_compatible.py
│       │
│       ├── ingestion/
│       │   ├── schema.py
│       │   ├── loader.py
│       │   ├── dense_embedding.py
│       │   ├── sparse_embedding.py
│       │   └── qdrant_ingest.py
│       │
│       ├── evaluation/
│       │   ├── ir_metrics.py
│       │   ├── ir_eval.py
│       │   ├── rag_eval.py
│       │   └── regression.py
│       │
│       ├── observability/
│       │   ├── tracing.py
│       │   ├── metrics.py
│       │   └── logging.py
│       │
│       ├── config/
│       │   ├── settings.py
│       │   └── constants.py
│       │
│       └── common/
│           ├── exceptions.py
│           └── utils.py
│
├── scripts/
│   ├── smoke_retrieval.py
│   ├── smoke_rag.py
│   ├── ingest_qdrant.py
│   ├── evaluate_ir.py
│   ├── evaluate_rag.py
│   ├── benchmark_retrieval.py
│   └── benchmark_rag.py
│
├── configs/
│   ├── retrieval.yaml
│   ├── index.yaml
│   ├── rag.yaml
│   ├── models.yaml
│   └── production.yaml
│
├── mlops/
│   ├── build_candidate_index.py
│   ├── evaluate_candidate.py
│   ├── promote_index.py
│   ├── rollback_index.py
│   └── pipeline_manifest.py
│
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       └── dashboards/
│
├── docker/
│   ├── rag-api.Dockerfile
│   └── qdrant/
│       └── docker-compose.yml
│
├── deploy/
│   ├── gpu-ec2/
│   │   ├── docker-compose.yml
│   │   └── env.example
│   │
│   └── qdrant-ec2/
│       ├── docker-compose.yml
│       └── env.example
│
├── tests/
│   ├── unit/
│   │   ├── test_fusion.py
│   │   ├── test_context_builder.py
│   │   ├── test_citation.py
│   │   └── test_config.py
│   │
│   ├── integration/
│   │   ├── test_qdrant.py
│   │   ├── test_retrieval.py
│   │   └── test_llm_client.py
│   │
│   └── regression/
│       └── test_golden_queries.py
│
├── data/
│   └── eval/
│       ├── ir_eval.jsonl
│       └── rag_eval.jsonl
│
├── artifacts/
│   └── .gitkeep
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
│
├── pyproject.toml
├── docker-compose.dev.yml
├── .env.example
├── .gitignore
├── README.md
└── plan.md