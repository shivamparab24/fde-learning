from typing import Any, Dict, List
import tiktoken

class ContextAssembler:
    def __init__(
        self, 
        max_total_tokens: int = 128000, 
        reserve_output_tokens: int = 4096,
        model_name: str = "gpt-4"
    ):
        self.max_total_tokens = max_total_tokens
        self.reserve_output_tokens = reserve_output_tokens
        self.input_budget = max_total_tokens - reserve_output_tokens
        
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except Exception:
            self.encoder = None

    def count_tokens(self, text: str) -> int:
        """Counts tokens using tiktoken or falls back to standard character estimation."""
        if self.encoder:
            return len(self.encoder.encode(text))
        return max(1, len(text) // 4)

    def assemble(
        self,
        system_prompt: str,
        user_query: str,
        history: List[Dict[str, str]],
        retrieved_docs: List[Dict[str, Any]],
        summary: str = "",
        history_ratio: float = 0.4
    ) -> Dict[str, Any]:
        """
        Assembles context prioritizing System Prompt -> User Query -> Running Summary -> History Window -> RAG Context.
        """
        remaining_budget = self.input_budget
        
        # --- PHASE 1: Mandatory Components ---
        sys_tokens = self.count_tokens(system_prompt)
        query_tokens = self.count_tokens(user_query)
        mandatory_total = sys_tokens + query_tokens + 8
        
        if mandatory_total > remaining_budget:
            raise ValueError(f"Mandatory elements ({mandatory_total} tokens) exceed total budget ({remaining_budget}).")
            
        remaining_budget -= mandatory_total

        # --- PHASE 1.5: Running Summary (if present) ---
        summary_clean = summary.strip() if summary else ""
        summary_tokens = self.count_tokens(summary_clean) + 6 if summary_clean else 0
        if summary_clean and summary_tokens <= remaining_budget:
            remaining_budget -= summary_tokens
        else:
            summary_tokens = 0
            summary_clean = ""
        
        # --- PHASE 2: Running Windowed History ---
        history_budget = int(remaining_budget * history_ratio)
        assembled_history = []
        trimmed_history = []
        history_tokens_used = 0
        
        for i, message in enumerate(reversed(history)):
            msg_tokens = self.count_tokens(message["content"]) + 4
            if history_tokens_used + msg_tokens <= history_budget:
                assembled_history.insert(0, message)
                history_tokens_used += msg_tokens
            else:
                # Remaining older messages are trimmed
                trimmed_history = history[: len(history) - i]
                break
                
        remaining_budget -= history_tokens_used
        
        # --- PHASE 3: Retrieved Context (Lost-in-the-Middle Mitigation) ---
        sorted_docs = sorted(retrieved_docs, key=lambda x: x.get("relevance", 0.0), reverse=True)
        assembled_docs = []
        rag_tokens_used = 0
        
        for doc in sorted_docs:
            doc_text = doc["text"]
            doc_tokens = self.count_tokens(doc_text) + 2
            if rag_tokens_used + doc_tokens <= remaining_budget:
                assembled_docs.append(doc_text)
                rag_tokens_used += doc_tokens
            else:
                break
                
        remaining_budget -= rag_tokens_used

        # Embed running summary & retrieved context into system payload
        summary_block = f"\n\n[RUNNING SUMMARY OF PAST CONVERSATION]\n{summary_clean}" if summary_clean else ""
        context_block = "\n\n".join(assembled_docs) if assembled_docs else "No external context retrieved."
        full_system_content = f"{system_prompt}{summary_block}\n\n[RETRIEVED CONTEXT / MEMORIES]\n{context_block}"

        final_messages = [{"role": "system", "content": full_system_content}]
        final_messages.extend(assembled_history)
        final_messages.append({"role": "user", "content": user_query})

        return {
            "messages": final_messages,
            "trimmed_history": trimmed_history,
            "telemetry": {
                "total_input_budget": self.input_budget,
                "used_input_tokens": self.input_budget - remaining_budget,
                "remaining_tokens": remaining_budget,
                "system_tokens": sys_tokens,
                "summary_active": bool(summary_clean),
                "summary_tokens": summary_tokens,
                "history_messages_count": len(assembled_history),
                "history_tokens": history_tokens_used,
                "trimmed_messages_count": len(trimmed_history),
                "docs_included_count": len(assembled_docs),
                "rag_tokens": rag_tokens_used
            }
        }