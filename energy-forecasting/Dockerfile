FROM python:3.13-slim
RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ ./src/

WORKDIR /home/appuser
RUN useradd -m appuser && chown -R appuser:appuser /home/appuser
USER appuser

EXPOSE ${PORT:-8080}

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8080}"]