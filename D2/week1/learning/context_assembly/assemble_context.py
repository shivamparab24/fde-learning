def assemble_context(system_instruction: str, retrieved_docs: list, user_query: str) -> str:
    """
    Assembles a structured prompt template segregating components clearly.
    """
    prompt_segments = []
    
    # 1. Top of the context window: System instructions and core behavior
    prompt_segments.append("<system_instructions>")
    prompt_segments.append(system_instruction.strip())
    prompt_segments.append("</system_instructions>\n")
    
    # 2. Middle of the context window: Reference documents
    prompt_segments.append("<reference_documents>")
    
    for i, doc in enumerate(retrieved_docs, 1):
        prompt_segments.append(f'<document id="{i}">\n{doc.strip()}\n</document>')
    prompt_segments.append("</reference_documents>\n")
    
    # 3. Bottom of the context window: Immediate task, query, and output marker
    prompt_segments.append("<user_query>")
    prompt_segments.append(user_query.strip())
    prompt_segments.append("</user_query>\n")
    
    prompt_segments.append("Assistant response:")
    
    return "\n".join(prompt_segments)

# # Example Usage
# sys_rule = "You are a precise data analysis assistant. Answer using only provided facts."
# docs = [
#     "Document 1: The project deadline is scheduled for October 14th.",
#     "Document 2: The assigned budget for Q3 is $45,000."
# ]
# query = "What is the Q3 budget?"

# assembled_prompt = assemble_context(sys_rule, docs, query)
# print(assembled_prompt)