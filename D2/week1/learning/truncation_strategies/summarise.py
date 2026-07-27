# from learning.truncation_strategies.drop import drop_truncate


# def summarize_chunk(text_chunk: str, low_cost_client) -> str:
#     """Uses a fast, cheap model to distill a text chunk."""
#     response = low_cost_client.chat.completions.create(
#         model="flash-model",
#         messages=[
#             {"role": "system", "content": "Condense this text into core facts. Keep it under 50 words."},
#             {"role": "user", "content": text_chunk}
#         ]
#     )
#     return response.choices[0].message.content

# def budget_by_summarization(text: str, max_tokens: int, tokenizer) -> str:
#     """Iteratively chunks and summarizes until the budget is met."""
#     tokens = tokenizer.encode(text)
#     if len(tokens) <= max_tokens:
#         return text

#     # Split into logical paragraphs or structural chunks
#     chunks = text.split("\n\n")
#     summarized_chunks = [summarize_chunk(c) for c in chunks]
#     combined = "\n\n".join(summarized_chunks)
    
#     # Recursion check if still over budget
#     if len(tokenizer.encode(combined)) > max_tokens:
#         return drop_truncate(combined, max_tokens * 4) # Fallback to hard drop
#     return combined


# Simulated cheap/fast LLM API call for demonstration purposes
def call_mini_llm_summary(text_to_compress: str) -> str:
    """Simulates a fast compression call to an LLM."""
    # In production, you would use: return client.chat.completions.create(...)
    return f"[Summary: {text_to_compress[:35]}...]"

def implement_summarize_strategy(data_list: list, max_budget: int) -> list:
    """ Compresses overflowing elements into a single summary block 
    
    so the total list length stays within the budget.
    """
    if len(data_list) <= max_budget:
        return data_list
        
    # Identify how many items are forcing us over budget
    overflow_count = len(data_list) - max_budget + 1
    items_to_compress = data_list[:overflow_count]
    items_to_keep = data_list[overflow_count:]
    
    # Merge and distill the overflow items
    raw_overflow_text = " | ".join(items_to_compress)
    distilled_summary = call_mini_llm_summary(raw_overflow_text)
    
    # Reconstruct the list: Summary block + the preserved items
    return [distilled_summary] + items_to_keep

# --- Execution Example ---
logs = [
    "Error: Database timeout at 10:00",
    "Error: Retry attempt 1 failed at 10:01",
    "Error: Retry attempt 2 failed at 10:02",
    "Status: Diverted traffic to replica at 10:03"
]
BUDGET = 2

result = implement_summarize_strategy(logs, BUDGET)
print("\n--- SUMMARIZE STRATEGY RESULT ---")
print(result)