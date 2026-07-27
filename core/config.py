from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):

    #environment variables
    ollama_base_url: str
    ollama_model_name: str
    groq_api_key: str
    groq_model_name: str

    model_config = SettingsConfigDict(
        env_file=".env"
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()