"""Configuration management using pydantic-settings."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration
    openai_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="OpenAI-compatible API base URL",
    )
    openai_api_key: str = Field(
        default="",
        description="API key for the LLM service",
    )
    openai_model: str = Field(
        default="Qwen2.5-Coder-7B-Instruct",
        description="Model name to use",
    )

    # RAG Configuration
    top_k: int = Field(default=8, description="Number of top results to retrieve")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for embeddings",
    )

    # Diff Guard
    diff_max_lines: int = Field(default=250, description="Max changed lines in a patch")
    diff_max_files: int = Field(default=6, description="Max files touched in a patch")

    # Optional Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant URL")
    qdrant_collection: str = Field(default="ggf_codebase", description="Qdrant collection name")

    # Retrieval Strategy
    retrieval_strategy: str = Field(
        default="keyword",
        description="Retrieval strategy: keyword, bm25, embedding, hybrid",
    )
    chunk_strategy: str = Field(
        default="fixed",
        description="Chunking strategy: fixed, ast, hybrid",
    )
    reranker_enabled: bool = Field(default=False, description="Enable cross-encoder reranking")

    # Fine-tuning
    finetune_model: str = Field(
        default="",
        description="Fine-tuned model ID (empty = use base model)",
    )

    # Experiment
    experiment_runs: int = Field(default=5, description="Number of runs per experiment variant")
    chain_of_thought: bool = Field(default=False, description="Enable chain-of-thought prompting")

    # Eval
    eval_timeout_seconds: int = Field(default=120, description="Timeout per task")
    log_level: str = Field(default="INFO", description="Logging level")

    # Paths (computed)
    repo_root: Path = Field(default=Path("."), description="Repository root path")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def get_settings(**overrides: object) -> Settings:
    """Create settings instance with optional overrides."""
    return Settings(**overrides)
