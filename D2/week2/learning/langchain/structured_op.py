from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from core.config import get_settings

settings = get_settings()

class UserProfile(BaseModel):
    name: str = Field(description="User's full name")
    age: int = Field(description="User's age")
    skills: list[str] = Field(description="List of technical skills")

llm = ChatOllama(model=settings.ollama_model_name, base_url=settings.ollama_base_url)

structured_llm = llm.with_structured_output(UserProfile)

response = structured_llm.invoke(
    "Extract info: Alex Mercer is a 29-year-old software engineer skilled in Python, Docker, and LangChain."
)

print(type(response))  # <class '__main__.UserProfile'>
print(f"Name: {response.name}, Skills: {response.skills}")