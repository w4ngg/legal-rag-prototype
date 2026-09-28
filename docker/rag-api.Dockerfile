# syntax=docker/dockerfile:1.7

# Build for the g4dn architecture with: docker build --platform linux/amd64 ...
# Override this build argument only with a reviewed CUDA/PyTorch image.
ARG PYTORCH_BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
ARG LEGAL_RAG_TARGET_PLATFORM=linux/amd64
FROM --platform=${LEGAL_RAG_TARGET_PLATFORM} ${PYTORCH_BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/var/cache/legal-rag/huggingface \
    HUGGINGFACE_HUB_CACHE=/var/cache/legal-rag/huggingface/hub \
    LEGAL_RAG_CONFIG=/etc/legal-rag/runtime.yaml

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin legalrag \
    && mkdir -p /etc/legal-rag /var/cache/legal-rag/huggingface \
    && chown -R legalrag:legalrag /var/cache/legal-rag

USER legalrag

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

# One worker is intentional: every worker would load a separate dense encoder
# and cross-encoder into GPU memory. Mount the production runtime YAML at
# /etc/legal-rag/runtime.yaml and the sparse-query bundle at the absolute path
# referenced by that YAML; neither secrets nor environment-specific endpoints
# are baked into this image.
CMD ["legal-rag-api", "--config", "/etc/legal-rag/runtime.yaml", "--host", "0.0.0.0", "--port", "8000", "--require-cuda"]
