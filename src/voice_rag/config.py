from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    sarvam_api_key: str | None = None
    sarvam_model: str = "saaras:v3"
    sarvam_mode: str = "codemix"
    sarvam_language_code: str = "unknown"

    answer_mode: str = "extractive"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"

    index_dir: Path = Path("artifacts/index")
    embedder: str = "hash"
    embedding_model: str = "intfloat/multilingual-e5-small"
    rag_top_k: int = 5
    rag_min_score: float = 0.18
    request_timeout_seconds: float = 12.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
