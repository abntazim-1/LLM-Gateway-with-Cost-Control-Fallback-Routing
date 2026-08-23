# syntax=docker/dockerfile:1
# Operator dashboard. A separate image from the gateway because they are
# independent processes with different lifecycles — the dashboard can be
# redeployed, scaled to zero, or omitted entirely without touching the API.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Both packages are required at build time even though only the dashboard
# runs here — the wheel is built from `gateway` and `dashboard` together.
COPY pyproject.toml README.md ./
COPY gateway/ gateway/
COPY dashboard/ dashboard/
RUN pip install --no-cache-dir .

# Which gateway to talk to. Overridden per environment; the default only
# makes sense when both run on the same host.
ENV GATEWAY_URL=http://localhost:8080

ENV PORT=8501
EXPOSE 8501

# headless suppresses the browser-opening and telemetry prompts, which
# otherwise block startup in a container. Shell form so $PORT expands.
CMD streamlit run dashboard/app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
