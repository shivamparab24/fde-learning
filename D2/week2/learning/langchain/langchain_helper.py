from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.config import get_settings

settings = get_settings()

MODEL = settings.ollama_model_name
BASE_URL = settings.ollama_base_url


# def generate_pet_name(animal_type: str):
#     llm = ChatOllama(
#         model=MODEL,
#         base_url=BASE_URL,
#         temperature=0.0
#     )

#     prompt_template = PromptTemplate(
#         input_variables=['animal_type'],
#         template="I have a pet {animal_type}, and i want a cool a name for it. Can you suggest me a single name? Only name."
#     )

#     # response = llm.invoke("I have a pet dog, and i want a cool a name for it. Can you suggest me a single name? Only name.")

#     name_chain = prompt_template | llm | StrOutputParser()

#     response = name_chain.invoke({"animal_type": animal_type})

#     return response

#     # print(response.content)
#     # print(response.response_metadata)
#     # print(response.usage_metadata)

#     # return response.content 
#     # return name # returns AI message from LLM -- langchain_core.messages.ai.AIMessage


# print(generate_pet_name("cat"))





llm = ChatOllama(
    model=MODEL,
    base_url=BASE_URL,
    temperature=0.7
)

prompt_template = PromptTemplate(
    input_variables=['animal_type', 'pet_color'],
    template="I have a {pet_color} pet {animal_type}, and I want a cool name for it. Can you suggest me 3 names?."
)

name_chain = prompt_template | llm | StrOutputParser()



def generate_pet_name(animal_type: str, pet_color: str):
    response = name_chain.invoke({"pet_color": pet_color, "animal_type": animal_type})
    return response

# print(generate_pet_name("green", "cat"))



if __name__ == "__main__":
    print("=== Welcome to the Pet Name Generator ===")

    animal_type = (
        input("Enter animal type (e.g., dog, cat, cow): ").strip().lower()
    )
    pet_color = input("Enter pet color (e.g., black, brown, green): ").strip()

    print("\nGenerating names...\n")
    suggestions = generate_pet_name(animal_type, pet_color)

    print("--- Suggested Names ---")
    print(suggestions)