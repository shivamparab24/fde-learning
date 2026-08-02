# def main():
#     print("Hello from code!")


# if __name__ == "__main__":
#     main()


from ollama import Client
from core.config import get_settings

settings = get_settings()

client = Client(host=settings.ollama_base_url)

response = client.chat(
    model=settings.ollama_model_name,
    messages=[{'role': 'user', 'content': 'Hello!'}],
)
print(response.message.content)