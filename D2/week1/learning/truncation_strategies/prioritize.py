# from typing import List, Dict

# def prioritize_and_pack(items: List[Dict], max_tokens: int, tokenizer) -> List[Dict]:
#     """Sorts items by priority score and packs up to the token budget."""
#     # 1. Sort by priority score descending
#     sorted_items = sorted(items, key=lambda x: x['priority_score'], reverse=True)
    
#     packed_items = []
#     current_tokens = 0
    
#     for item in sorted_items:
#         # Estimate or calculate tokens for the current item structure
#         item_text = f"Source: {item['source']}\nContent: {item['content']}\n"
#         item_tokens = len(tokenizer.encode(item_text))
        
#         if current_tokens + item_tokens <= max_tokens:
#             packed_items.append(item)
#             current_tokens += item_tokens
#         else:
#             # Budget full. We gracefully skip lower priority items.
#             continue
            
#     return packed_items



def implement_prioritize_strategy(document_chunks: list, max_word_budget: int) -> list:
    """ Sorts elements by a priority score and packs them into the 
    
    budget window until no more items fit.
    """
    # 1. Sort elements by priority score in descending order (highest value first)
    sorted_by_value = sorted(document_chunks, key=lambda x: x['score'], reverse=True)
    
    packed_bag = []
    current_word_count = 0
    
    # 2. Pack the bag greedily
    for chunk in sorted_by_value:
        chunk_words = len(chunk['text'].split())
        
        # Check if this item fits in our remaining word budget
        if current_word_count + chunk_words <= max_word_budget:
            packed_bag.append(chunk)
            current_word_count += chunk_words
        else:
            # Skip this item if it's too large, but keep checking subsequent smaller items
            continue
            
    return packed_bag

# --- Execution Example ---
# A RAG system returns 3 chunks with varying relevance scoresl
knowledge_base_chunks = [
    {"doc_id": "A", "text": "Python uses timsort for sorting arrays efficiently.", "score": 0.45},
    {"doc_id": "B", "text": "The budget strategy is critical for LLM context windows.", "score": 0.92},
    {"doc_id": "C", "text": "Coffee consumption peaks at 9 AM globally.", "score": 0.12}
]
WORD_BUDGET = 10 # Strict budget that cannot hold all text

result = implement_prioritize_strategy(knowledge_base_chunks, WORD_BUDGET)
print("\n--- PRIORITIZE STRATEGY RESULT ---")
for item in result:
    print(f"Packed Doc {item['doc_id']} (Score: {item['score']}): {item['text']}")
# Output will only contain Doc B because it has the highest relevance score and fills the budget.