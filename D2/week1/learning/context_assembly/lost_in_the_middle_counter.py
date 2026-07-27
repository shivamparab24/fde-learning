def inside_out_rerank(scored_docs: list) -> list:
    """
    Sorts documents such that the most relevant items are placed at 
    the boundaries (beginning and end) of the prompt context.
    
    Expects a list of tuples or dicts containing: (document_text, score)
    Sorted assuming higher score = higher relevance.
    """
    # Sort by score descending (highest relevance first)
    sorted_by_relevance = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    
    reranked = [None] * len(sorted_by_relevance)
    
    left = 0
    right = len(sorted_by_relevance) - 1
    
    # Distribute documents to the edges alternatingly
    for i, doc_tuple in enumerate(sorted_by_relevance):
        if i % 2 == 0:
            reranked[left] = doc_tuple[0]
            left += 1
        else:
            reranked[right] = doc_tuple[0]
            right -= 1
            
    return reranked

# Example Usage
retrieved_data = [
    ("Irrelevant noise document text.", 0.12),
    ("Highly critical target information.", 0.98),
    ("Semi-relevant background information.", 0.45),
    ("Moderately important supporting data.", 0.76)
]

# Standard sorting places lowest score at the end.
# Inside-out sorting moves top scores to the outer edges:
ordered_docs = inside_out_rerank(retrieved_data)
for idx, doc in enumerate(ordered_docs):
    print(f"Position {idx}: {doc[:40]}...")