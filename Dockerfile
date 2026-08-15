FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY web ./web
COPY data ./data
COPY scripts ./scripts

RUN pip install --no-cache-dir . \
    && python scripts/build_demo_index.py

EXPOSE 8000
CMD ["uvicorn", "voice_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
