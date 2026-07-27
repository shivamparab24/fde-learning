import json
import time
import requests

from core.config import get_settings
from .lost_in_middle import generate_lost_in_the_middle_experiment

settings = get_settings()

# 1. Configuration for your remote PC
OLLAMA_URL = settings.ollama_base_url
MODEL_NAME = settings.ollama_model_name

# Test positions across the text payload: 0% (start) to 100% (end)
positions_to_test = [0.0, 0.25, 0.5, 0.75, 1.0]
# positions_to_test = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
results = {}

print(f"Connecting directly to Ollama endpoint: {OLLAMA_URL}")
print(f"Running native API analysis on: {MODEL_NAME}\n")

for pos in positions_to_test:
    # Generate the payload list of message dicts from your function
    payload_messages = generate_lost_in_the_middle_experiment(position=pos)
    
    # Structure the raw HTTP JSON payload for Ollama's API
    data = {
        "model": MODEL_NAME,
        "messages": payload_messages,
        "stream": False,  # Return the full answer at once instead of token-by-token
        "options": {
            "temperature": 0.0  # Turn off randomness for strict testing
        }
    }
    
    start_time = time.time()
    
    try:
        # Send the POST request directly to the remote PC
        response = requests.post(OLLAMA_URL, json=data, timeout=600)
        response.raise_for_status()  # Catch HTTP errors
        
        # Parse Ollama's native JSON response structure
        response_json = response.json()
        answer = response_json["message"]["content"]
        elapsed = time.time() - start_time
        
        # Verify if the model successfully extracted the token key
        is_correct = "BLUE_MAGPIE_99" in answer
        
        if is_correct:
            results[pos] = f"SUCCESS ({elapsed:.1f}s)"
        else:
            results[pos] = f"FAILED ({elapsed:.1f}s) -> Model: '{answer.strip()[:50]}...'"
            
    except requests.exceptions.RequestException as e:
        results[pos] = f"NETWORK/API ERROR: {e}"
    
    print(f"Position {pos*100:>3.0f}% Depth: {results[pos]}")

print("\n=== FINAL NATIVE OLLAMA RESULTS ===")
for pos, res in results.items():
    print(f"Needle location at {pos*100:3.0f}% depth -> {res}")