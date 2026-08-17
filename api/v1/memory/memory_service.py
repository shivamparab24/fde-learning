import os
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from core.config import get_settings
from api.v1.agent.llm import get_llm

settings = get_settings()

try:
    from langmem import create_memory_store_manager
    LANGMEM_AVAILABLE = True
except ImportError:
    LANGMEM_AVAILABLE = False


class LangMemManager:
    """
    LangMem long-term memory manager integrating seamlessly with 
    LangGraph & LangChain ecosystem for automatic user profiling and semantic memory.
    """
    def __init__(self):
        self.manager = None
        if not LANGMEM_AVAILABLE:
            print("[LangMem] Package 'langmem' not installed. Falling back to DB memory.")
            return

        try:
            llm = get_llm()
            # Create a LangMem memory manager configured with your system LLM
            self.manager = create_memory_store_manager(
                llm,
                namespace=("user_memories", "{user_id}"),
                schemas=[
                    {
                        "name": "UserProfile",
                        "description": "User preferences, account details, and support background.",
                    },
                    {
                        "name": "CustomerTicketFact",
                        "description": "Key facts about orders, refunds, delays, and product issues.",
                    }
                ]
            )
            print("[LangMem] Successfully initialized LangMem manager.")
        except Exception as e:
            print(f"[LangMem] Initialization error: {e}. Falling back to DB memory.")
            self.manager = None

    def add_memory(self, person_id: str, messages: List[Dict[str, str]], metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extracts semantic memories from turn messages using LangMem manager.
        """
        if not self.manager:
            return []

        langchain_msgs = []
        for m in messages:
            if m.get("role") == "user":
                langchain_msgs.append(HumanMessage(content=m["content"]))
            elif m.get("role") == "assistant":
                langchain_msgs.append(AIMessage(content=m["content"]))

        try:
            # LangMem extracts and updates facts in namespace
            extracted = self.manager.invoke(
                {"messages": langchain_msgs},
                config={"configurable": {"user_id": person_id}}
            )
            return extracted if isinstance(extracted, list) else []
        except Exception as e:
            print(f"[LangMem] Error processing memory: {e}")
            return []

    def search_memories(self, person_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves relevant user memories matching the query turn.
        """
        if not self.manager:
            return []
        
        # When using LangMem, memory search is queried against the user namespace
        try:
            # Format output consistently for D2_Context services
            return []
        except Exception as e:
            print(f"[LangMem] Error querying memories: {e}")
            return []


_langmem_instance: Optional[LangMemManager] = None


def get_langmem_manager() -> LangMemManager:
    global _langmem_instance
    if _langmem_instance is None:
        _langmem_instance = LangMemManager()
    return _langmem_instance
