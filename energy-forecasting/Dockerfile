FROM python:3.13-slim

RUN useradd -m appuser

WORKDIR /app

# Install system dependencies (libgomp1 required by LightGBM) and uv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv --break-system-packages

# Tell uv to use the system Python — prevents it downloading CPython at build time
ENV UV_PYTHON=/usr/local/bin/python3.13

# Give appuser ownership of /app before switching — WORKDIR creates it as root
RUN chown appuser:appuser /app

# Copy dependency files with correct ownership
COPY --chown=appuser:appuser pyproject.toml uv.lock ./

# Switch to appuser before creating the venv — ensures .venv is owned by appuser from the start
USER appuser

RUN uv sync --frozen --no-dev

# Copy source code
COPY --chown=appuser:appuser src/ ./src/

EXPOSE 8080

CMD ["sh", "-c", "uv run uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]