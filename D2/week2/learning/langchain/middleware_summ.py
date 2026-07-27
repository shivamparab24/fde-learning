from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from core.config import get_settings

settings = get_settings()

@tool
def calculate_tax(income: float) -> str:
    """Calculate basic income tax."""
    return f"Estimated tax for income ${income}: ${income * 0.20:.2f}"

llm = ChatOllama(model=settings.ollama_model_name, base_url=settings.ollama_base_url)

agent = create_agent(
    model=llm,
    tools=[calculate_tax],
    system_prompt="You are a helpful financial assistant.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "How much tax do I owe if I make 85000?"}]}
)

print(result["messages"][-1].content)