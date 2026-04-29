FROM python:3.11-slim

# Copy dependency files first (cached)
COPY pyproject.toml uv.lock ./

# Install dependencies with pip (simpler than uv for Docker)
RUN pip install --no-cache-dir -r pyproject.toml

# Copy source code
COPY src/ ./src/

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /home/appuser
USER appuser

EXPOSE ${PORT:-8080}

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8080}"]