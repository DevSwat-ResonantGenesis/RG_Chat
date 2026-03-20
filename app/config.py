import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    SERVICE_NAME: str = "chat_service"

    POSTGRES_HOST: str = "chat_db"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "chat_user"
    POSTGRES_PASSWORD: str = "chat_pass"
    POSTGRES_DB: str = "chat_db"

    # API Keys for LLM providers
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None

    # Service URLs
    LLM_SERVICE_URL: str = "http://llm_service:8000"
    MEMORY_SERVICE_URL: str = "http://memory_service:8000"

    class Config:
        env_file = ".env"
        env_prefix = "CHAT_"
        case_sensitive = False


settings = Settings()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    os.getenv(
        "CHAT_DATABASE_URL",
        f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )
).replace("postgresql://", "postgresql+asyncpg://").replace("?sslmode=", "?ssl=")
