messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    
    # First turn
    {"role": "user", "content": "How do I fix a 'KeyError' in Python?"},
    {"role": "assistant", "content": "Use the '.get()' method or check if the key exists before accessing it."},
    
    # Second turn
    {"role": "user", "content": "Thanks. What is a quick dinner idea?"},
    {"role": "assistant", "content": "Garlic chili oil noodles. Just toss boiled noodles with soy sauce, chili oil, and garlic."},
    
    # Third turn (relies on context)
    {"role": "user", "content": "Can you add a protein to that?"}
]