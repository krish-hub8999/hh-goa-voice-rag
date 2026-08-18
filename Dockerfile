FROM python:3.11-slim

# 1. Install system binaries required for audio handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy configuration & source code
COPY pyproject.toml README.md ./
COPY src ./src
COPY web ./web
COPY data ./data
COPY scripts ./scripts

# 3. Install Python dependencies and build index
RUN pip install --no-cache-dir . \
    && python scripts/build_demo_index.py

# 4. Support dynamic Render port binding with local fallback
ENV PORT=8000
EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn voice_rag.api:app --host 0.0.0.0 --port ${PORT}"]
