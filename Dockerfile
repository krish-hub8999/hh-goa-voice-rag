FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY web ./web
COPY data ./data

RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "voice_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
