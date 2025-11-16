"""
Configuration management for FinSearch-AI.
Uses Pydantic for validation and YAML for configuration files.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class DataConfig(BaseModel):
    """Data processing configuration"""
    raw_path: str = "data/raw"
    interim_path: str = "data/interim"
    processed_path: str = "data/processed"
    chunk_size: int = 512
    chunk_overlap: int = 128


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration"""
    model: str = "BAAI/bge-small-en-v1.5"
    dimension: int = 384
    batch_size: int = 32


class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    use_hybrid: bool = True
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    top_k: int = 20

    @validator('dense_weight', 'sparse_weight')
    def validate_weights(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Weights must be between 0 and 1')
        return v


class RerankingConfig(BaseModel):
    """Reranking configuration"""
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    top_k: int = 5


class GenerationConfig(BaseModel):
    """LLM generation configuration"""
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 1000

    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('Temperature must be between 0 and 2')
        return v


class EvaluationConfig(BaseModel):
    """Evaluation configuration"""
    metrics: List[str] = ["precision_at_k", "recall_at_k", "mrr", "ndcg"]
    k_values: List[int] = [1, 3, 5, 10, 20]


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class Settings(BaseSettings):
    """Main settings class combining all configurations"""

    # Sub-configurations
    data: DataConfig = Field(default_factory=DataConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # API settings (for serving)
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_reload: bool = False

    # Environment settings
    environment: str = "development"
    debug: bool = False

    class Config:
        env_prefix = "FINSEARCH_"
        env_nested_delimiter = "__"
        case_sensitive = False


def load_config(config_path: Optional[str] = None) -> Settings:
    """
    Load configuration from YAML file and environment variables.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Settings object with loaded configuration
    """
    # Default config path
    if config_path is None:
        config_path = os.getenv("FINSEARCH_CONFIG", "configs/default.yaml")

    config_dict = {}

    # Load from YAML if file exists
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}

    # Create settings (will also load from environment variables)
    settings = Settings(**config_dict)

    # Setup logging based on configuration
    setup_logging(settings.logging)

    return settings


def setup_logging(logging_config: LoggingConfig):
    """Setup logging based on configuration"""
    import logging

    # Convert string level to logging constant
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    level = level_map.get(logging_config.level.upper(), logging.INFO)

    # Configure logging
    logging.basicConfig(
        level=level,
        format=logging_config.format
    )


def save_config(settings: Settings, output_path: str):
    """
    Save configuration to YAML file.

    Args:
        settings: Settings object to save
        output_path: Path to save the configuration
    """
    # Convert to dictionary
    config_dict = settings.model_dump()

    # Save to YAML
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence.

    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    import copy

    merged = copy.deepcopy(base_config)

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


# Singleton instance for global access
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    if _settings is None:
        _settings = load_config()
    return _settings


def reset_settings():
    """Reset global settings instance"""
    global _settings
    _settings = None