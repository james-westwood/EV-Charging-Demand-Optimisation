# Install uv and use it to install deps (simpler than fighting with pip+pyproject.toml)
FROM python:3.13-slim
RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -e .

COPY src/ ./src/

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /home/appuser
USER appuser

EXPOSE ${PORT:-8080}

CMD [".venv/bin/uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8080}"]