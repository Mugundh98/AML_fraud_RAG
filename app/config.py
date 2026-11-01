from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-latest"
    # Tried before Groq: a separate capacity pool, so it often works when Flash returns 503.
    gemini_fallback_model: str = "gemini-flash-lite-latest"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.1
    llm_timeout_seconds: int = 30

    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "aml_docs"
    qdrant_local_path: Path = PROJECT_ROOT / "storage" / "qdrant"

    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 6

    neo4j_uri: str = ""
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    max_transactions: int = 50_000
    synthetic_dir: Path = PROJECT_ROOT / "data" / "synthetic"

    # Phase 4: outside services. Each MCP server is only started when its credentials are set.
    tavily_api_key: str = ""
    tavily_mcp_url: str = "https://mcp.tavily.com/mcp/"
    google_client_secret_file: Path = PROJECT_ROOT / "secrets" / "google_client_secret.json"
    google_token_file: Path = PROJECT_ROOT / "secrets" / "google_token.json"
    slack_bot_token: str = ""
    slack_default_channel: str = ""
    # Pending write actions (Slack posts, calendar events) expire if nobody approves them.
    approval_ttl_minutes: int = 30

    docs_dir: Path = PROJECT_ROOT / "data" / "docs"
    api_url: str = "http://localhost:8000"

    @field_validator("google_client_secret_file", "google_token_file")
    @classmethod
    def _relative_to_project(cls, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
