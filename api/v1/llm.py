from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from core.config import get_settings

settings = get_settings()


def get_llm():
    # _llm = ChatOllama(
    #     base_url=settings.ollama_base_url,
    #     model=settings.ollama_model_name,
    #     temperature=0.7,
    # )

    _llm = ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=settings.groq_model_name,
        temperature=0.7,
    )


    return _llm