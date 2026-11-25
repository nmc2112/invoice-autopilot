FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir ".[ui,local]"

# Download the embedding model at build time: no network call and no quota at runtime.
ENV FASTEMBED_CACHE_PATH=/app/models
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='/app/models')"

COPY config/ config/
COPY prompts/ prompts/
COPY app.py .

RUN useradd --create-home appuser && mkdir -p data && chown -R appuser data /app/models
USER appuser
EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn invoice_autopilot.api:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
