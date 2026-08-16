# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Oracle Ampere A1 is CPU-only. Install PyTorch/TorchVision explicitly from
# PyTorch's official CPU wheel index before the remaining dependencies.
# PyPI's generic Linux ARM64 torch wheel can pull CUDA/NVIDIA packages, which
# are unnecessary on A1 and can fail platform validation (for example
# nvidia-cusparselt). Keep the repository's single cumulative requirements.txt;
# the filtered file below exists only inside this Docker build.
ARG PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_CPU_VERSION=2.13.0+cpu
ARG TORCHVISION_CPU_VERSION=0.28.0+cpu

RUN python -m pip install --no-cache-dir --upgrade pip \
    && grep -Ev '^[[:space:]]*(torch|torchvision)([<>=!~[:space:]]|$)' /app/requirements.txt > /tmp/requirements-without-torch.txt \
    && python -m pip install --no-cache-dir \
         --index-url "${PYTORCH_CPU_INDEX_URL}" \
         --extra-index-url https://pypi.org/simple \
         "torch==${TORCH_CPU_VERSION}" \
         "torchvision==${TORCHVISION_CPU_VERSION}" \
    && python -m pip install --no-cache-dir -r /tmp/requirements-without-torch.txt \
    && python -m pip check \
    && rm -f /tmp/requirements-without-torch.txt

COPY alembic.ini /app/alembic.ini
COPY migrations /app/migrations
COPY src /app/src
COPY artifacts /app/artifacts

# Phase 10 Step 5A target. Build this target explicitly with
# --platform linux/arm64 before creating the Oracle Ampere A1 VM. The build
# intentionally contains no model weights, MRI studies, secrets or runtime
# caches; it verifies Linux ARM64 dependency/native-library compatibility and
# construction of the frozen model architectures only.
FROM runtime-base AS oracle-arm64-preflight
RUN python -m gbm_ai.deployment.oracle_arm64_preflight
CMD ["python", "-m", "gbm_ai.deployment.oracle_arm64_preflight"]

FROM runtime-base AS api
EXPOSE 8080
CMD ["sh", "-c", "python -m uvicorn gbm_ai.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
