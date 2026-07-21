import tiktoken
from typing import List, Dict, Any

class ContextAssembler:
    def __init__(
        self, 
        max_total_tokens: int = 128000, 
        reserve_output_tokens: int = 4096,
        model_name: str = "gpt-4"
    ):
        self.max_total_tokens = max_total_tokens
        self.reserve_output_tokens = reserve_output_tokens
        
        # Calculate strict input budget
        self.input_budget = max_total_tokens - reserve_output_tokens
        
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except Exception:
            # Fallback character-based token estimator
            self.encoder = None

    def count_tokens(self, text: str) -> int:
        """Accurately count tokens or fall back to an estimate."""
        if self.encoder:
            return len(self.encoder.encode(text))
        # Fallback: roughly 4 characters per English token
        return max(1, len(text) // 4)

    def assemble(
        self,
        system_prompt: str,
        user_query: str,
        history: List[Dict[str, str]],  # Format: [{"role": "user", "content": "..."}]
        retrieved_docs: List[Dict[str, Any]] # Format: [{"text": "...", "relevance": 0.95}]
    ) -> Dict[str, Any]:
        """
        Assembles the context dynamically enforcing rules of precedence.
        """
        remaining_budget = self.input_budget
        
        # --- PHASE 1: Mandatory Components (System + Current User Query) ---
        system_tokens = self.count_tokens(system_prompt)
        query_tokens = self.count_tokens(user_query)
        
        mandatory_total = system_tokens + query_tokens
        if mandatory_total > remaining_budget:
            raise ValueError(
                f"Mandatory prompt elements ({mandatory_total} tokens) "
                f"exceed input budget of {remaining_budget} tokens."
            )
            
        remaining_budget -= mandatory_total
        
        # --- PHASE 2: Assemble Conversation History (Newest First) ---
        # We process history in reverse chronological order to keep the most recent context
        assembled_history = []
        history_tokens_used = 0
        
        for turn in reversed(history):
            turn_tokens = self.count_tokens(turn["content"])
            # In a real chat API, roles and metadata consume small overhead.
            # We add 4 structural tokens per message as a safe buffer.
            total_turn_tokens = turn_tokens + 4 
            
            if history_tokens_used + total_turn_tokens <= (remaining_budget * 0.4): # Allocate max 40% of leftover to history
                assembled_history.insert(0, turn)
                history_tokens_used += total_turn_tokens
            else:
                break # Stop adding older history once budget/ratio is reached
                
        remaining_budget -= history_tokens_used
        
        # --- PHASE 3: Assemble Retrieved Context / RAG (Highest Relevance First) ---
        # Sort documents by relevance score descending
        sorted_docs = sorted(retrieved_docs, key=lambda x: x.get("relevance", 0), reverse=True)
        
        assembled_docs = []
        rag_tokens_used = 0
        
        for doc in sorted_docs:
            doc_text = doc["text"]
            doc_tokens = self.count_tokens(doc_text) + 2 # buffer
            
            if rag_tokens_used + doc_tokens <= remaining_budget:
                assembled_docs.append(doc_text)
                rag_tokens_used += doc_tokens
            else:
                # Optional: We could truncate the last fitting document, 
                # but keeping documents whole prevents context fragmentation.
                continue 
                
        remaining_budget -= rag_tokens_used

        # --- PHASE 4: Compile Final Payload ---
        # Format the final retrieved context block
        context_block = "\n\n".join(assembled_docs) if assembled_docs else "No relevant context found."
        
        final_messages = [
            {"role": "system", "content": f"{system_prompt}\n\n[Retrieved Context]\n{context_block}"}
        ]
        final_messages.extend(assembled_history)
        final_messages.append({"role": "user", "content": user_query})
        
        total_used = self.input_budget - remaining_budget
        
        return {
            "messages": final_messages,
            "metrics": {
                "total_input_tokens": total_used,
                "remaining_input_budget": remaining_budget,
                "system_tokens": system_tokens,
                "history_turns_included": len(assembled_history),
                "history_tokens": history_tokens_used,
                "docs_included": len(assembled_docs),
                "rag_tokens": rag_tokens_used,
                "reserved_output_tokens": self.reserve_output_tokens
            }
        }