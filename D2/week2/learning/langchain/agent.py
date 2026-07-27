from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from core.config import get_settings

settings = get_settings()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model = ChatOllama(
    model=settings.ollama_model_name,
    base_url=settings.ollama_base_url,
)

agent = create_agent(
    model=model,

    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content_blocks)