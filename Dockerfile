# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Fail fast and log immediately — buffered stdout hides startup errors in
# hosted log viewers, which is exactly when you need them.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# The wheel is built from both `gateway` and `dashboard` (see
# [tool.hatch.build.targets.wheel]), so both must be present before install.
# Copying only dependency metadata first — the usual layer-caching trick —
# fails here because the build backend needs the packages themselves.
COPY pyproject.toml README.md ./
COPY gateway/ gateway/
COPY dashboard/ dashboard/
RUN pip install --no-cache-dir .

COPY configs/ configs/

# Cloud backends by default: a container has no local Ollama to reach, so the
# repository's default config would start with every backend unreachable.
ENV BACKENDS_CONFIG_PATH=configs/backends.cloud.yaml

# The ledger lives here. On hosts with an ephemeral filesystem this resets on
# redeploy — fine for a demo, but mount a volume to keep spend history.
ENV LEDGER_DB_PATH=/app/data/ledger.db
RUN mkdir -p /app/data

# Hosts inject the port to listen on; 8080 is only the local default.
ENV PORT=8080
EXPOSE 8080

# Shell form so $PORT expands at runtime rather than being passed literally.
CMD uvicorn gateway.main:app --host 0.0.0.0 --port ${PORT}
