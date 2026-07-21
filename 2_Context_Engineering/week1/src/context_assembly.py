import tiktoken

def assemble_context(
    system_prompt: str,
    user_prompt: str,
    history: list[dict],
    retrieved_docs: list[str],
    max_budget: int = 128000,
    generation_reserve: int = 512,
    model: str = "gpt-4"
) -> list[dict]:
    """
    Assembles an LLM context payload within a strict token budget.
    Prioritizes: System > User Prompt > Chat History > Retrieved Docs.
    """
    encoder = tiktoken.encoding_for_model(model)
    
    # 1. Deduct target generation room immediately
    available_tokens = max_budget - generation_reserve
    
    # 2. Hard requirements (System & current User prompt)
    sys_tokens = len(encoder.encode(system_prompt))
    user_tokens = len(encoder.encode(user_prompt))
    
    # Base overhead estimation per message (~4 tokens for structural chat wrappers)
    current_used = sys_tokens + user_tokens + 8 
    
    if current_used > available_tokens:
        raise ValueError("System and User prompt exceed the entire budget limits.")
        
    # 3. Budget history (Newest first, truncate older if necessary)
    valid_history = []
    for msg in reversed(history):
        msg_tokens = len(encoder.encode(msg["content"])) + 4
        if current_used + msg_tokens <= (available_tokens - 100): # keeping a tiny buffer
            valid_history.insert(0, msg)
            current_used += msg_tokens
        else:
            break # Drop older history if budget is hit
            
    # 4. Budget & Rank Retrieved Docs
    # To mitigate 'lost in the middle', place the highest ranked items at the edges
    # of the allowed document space, or simply fill up to remaining tokens.
    valid_docs = []
    for doc in retrieved_docs:
        doc_tokens = len(encoder.encode(doc)) + 4
        if current_used + doc_tokens <= available_tokens:
            valid_docs.append(doc)
            current_used += doc_tokens
        else:
            # Truncation strategy: skip or summarize. Here we drop the lower priority doc.
            continue 
            
    context_str = "\n\n".join(valid_docs)
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(valid_history)
    
    # Inject retrieved context right before the user prompt
    if context_str:
        messages.append({"role": "user", "content": f"Retrieved Context:\n{context_str}"})
        
    messages.append({"role": "user", "content": user_prompt})
    
    print(f"Total tokens built into payload: {current_used}")
    return messages