from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from core.config import get_settings

settings = get_settings()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a tech mentor. Explain concepts in 2 bullet points for a beginner."),
    ("user", "What is {concept}?")
])

llm = ChatOllama(
    model=settings.ollama_model_name, 
    base_url=settings.ollama_base_url, 
    temperature=0.7,
)

parser = StrOutputParser()

# LCEL Chain
chain = prompt | llm | parser

result = chain.invoke({"concept": "Dependency Injection"})
print(result)