# Install uv first (changes rarely)
FROM python:3.11-slim as builder
RUN pip install uv

# Copy dependency files (cached until packages change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-interact --offline

# Copy feature store (changes on data collection runs)
COPY data/features/ ./data/features/

# Copy source code (changes most often)
COPY src/ ./src/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
RUN chown -R appuser:appuser /home/appuser
USER appuser

# Expose port from Cloud Run env var, fallback to 8080 for local dev
EXPOSE ${PORT:-8080}

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8080}"]