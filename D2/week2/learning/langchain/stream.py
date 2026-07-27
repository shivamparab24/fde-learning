from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from core.config import get_settings

settings = get_settings()

llm = ChatOllama(model=settings.ollama_model_name, base_url=settings.ollama_base_url)


for chunk in llm.stream([HumanMessage(content="Write a 3-line poem about debugging code.")]):
    print(chunk.content, end="", flush=True)
print()