import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()


class Settings:
    def __init__(self):
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        self.ollama_model_name = os.getenv("OLLAMA_MODEL_NAME")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model_name = os.getenv("GROQ_MODEL_NAME")


settings = Settings()
