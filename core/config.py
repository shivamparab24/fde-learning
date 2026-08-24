from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    
    model_config = SettingsConfigDict(
        env_file=".env"
    )

    #environment variables
    ollama_base_url: str
    ollama_model_name: str

    groq_api_key: str
    groq_model_name: str

    token_budget: int
    reserve_output_tokens: int

    database_url: str
    postgres_host: str
    postgres_port: str
    postgres_db: str
    postgres_user: str
    postgres_password: str
    
    embedding_model_name: str
    embedding_dims: int


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()