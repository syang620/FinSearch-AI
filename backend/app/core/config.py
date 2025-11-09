from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "FinSearch AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # API settings
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]

    # LLM settings
    LLM_MODEL_NAME: str = "google/flan-t5-base"  # Can switch to "mistralai/Mistral-7B-Instruct-v0.1"
    LLM_MAX_LENGTH: int = 512
    LLM_TEMPERATURE: float = 0.7
    LLM_DEVICE: str = "cpu"  # Change to "cuda" if GPU available

    # Embedding settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # ChromaDB settings
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "financial_documents"

    # Sentiment Analysis settings
    SENTIMENT_MODEL: str = "ProsusAI/finbert"

    # Document processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_DOCUMENT_SIZE_MB: int = 50

    # Database
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # Data ingestion
    DCF_EMAIL: Optional[str] = None
    DCF_PASSWORD: Optional[str] = None
    EDGAR_USER_EMAIL: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
