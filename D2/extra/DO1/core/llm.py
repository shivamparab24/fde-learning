from groq import Groq
import time
import requests

from DO1.core.config import settings


class OllamaLLM:

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model_name

    def generate(self, messages):

        start_time = time.time()
        response = requests.post(
            f"{self.base_url}",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False
            },
            timeout=700
        )
        processing_time = time.time() - start_time
        response.raise_for_status()
        result = response.json()

        return {
            "response": result["message"]["content"],
            "processing_time": processing_time
        }


class GroqLLM:

    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model_name

    def generate(self, prompt):

        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        total_time = time.perf_counter() - start_time

        return {
            "response": response.choices[0].message.content,
            "processing_time": total_time,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
