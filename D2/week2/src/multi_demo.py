from datetime import datetime
import os
import sqlite3
import sys
import time
import re

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables import RunnableConfig

from core.config import get_settings

settings = get_settings()


# llm = ChatGroq(
#     api_key=settings.groq_api_key,
#     model=settings.groq_model_name,
#     temperature=0.7,
#     keep_alive="1h",      
# )

llm = ChatOllama(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model_name,
    temperature=0.7,
    keep_alive="1h", # -1 for indefinite time
)

SYSTEM_PROMPT = ""

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def call_model(state: ChatState, config: RunnableConfig):  # Added: RunnableConfig parameter
    full_prompt = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    start = time.perf_counter()
    full_response = ""
    token_usage = {}

    # Added: Extract thread_id from config
    thread_id = config.get("configurable", {}).get("thread_id", "N/A")

    print("\nAssistant: ", end="", flush=True)
    
    for chunk in llm.stream(full_prompt):
        content = chunk.content
        full_response += content
        print(content, end="", flush=True)
        
        if chunk.usage_metadata:
            token_usage = chunk.usage_metadata

    latency = (time.perf_counter() - start) * 1000
    
    input_tokens = token_usage.get("input_tokens", "N/A")
    output_tokens = token_usage.get("output_tokens", "N/A")
    total_tokens = token_usage.get("total_tokens", "N/A")

    # Added: Print thread_id with metrics
    print(f"\n\n[Session] Thread ID: {thread_id}")
    print(f"[Metrics] Latency: {latency:.2f} ms")
    print(f"[Tokens]  Prompt: {input_tokens} | Completion: {output_tokens} | Total: {total_tokens}")
    print("-" * 50)
    
    return {"messages": [("assistant", full_response)]}

# Build LangGraph workflow
builder = StateGraph(ChatState)
builder.add_node("assistant", call_model)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

# checkpointer,  stores in RAM only
# graph = builder.compile(checkpointer=MemorySaver())

# SQLite checkpointer to keep state when script restarts
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
graph = builder.compile(checkpointer=SqliteSaver(conn))

# def main():
#     global SYSTEM_PROMPT

#     print("=" * 50)
#     user_sys_prompt = input("Enter System Prompt (or press ENTER for default): ").strip()
#     base_prompt = user_sys_prompt or "You are a helpful technical assistant."
    
#     # Large static context padding forces KV cache hit in Ollama
#     static_context = "".join(f"\n- Knowledge Base Guideline #{i}: Fast local execution context." for i in range(100))
#     SYSTEM_PROMPT = base_prompt + "\n" + static_context
#     print("=" * 50)

#     session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#     config = {"configurable": {"thread_id": session_id}}
#     print(f"Active Thread ID: {session_id}")
    
#     turn = 1
#     while True:
#         user_input = input(f"\n[Turn {turn}] You: ").strip()
#         if not user_input:
#             continue
#         if user_input.lower() in ["exit", "quit", "q"]:
#             break

#         graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
#         turn += 1

#     print("\nGoodbye!")

# if __name__ == "__main__":
#     main()


def list_existing_sessions(conn: sqlite3.Connection) -> list[str]:
    """Helper to fetch all saved thread_ids from the SQLite database."""
    cursor = conn.cursor()
    # LangGraph SqliteSaver stores checkpoints in the 'checkpoints' table
    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
    return [row[0] for row in cursor.fetchall()]

def main():
    print("=" * 50)
    existing_sessions = list_existing_sessions(conn)
    
    if existing_sessions:
        print("Existing Session IDs found:")
        for sid in existing_sessions:
            print(f" - {sid}")
        print("-" * 50)

    # 1. Get user input for session ID
    user_session_input = input("Enter Session ID to resume (or press ENTER to start a new session): ").strip()

    is_resuming = False
    if user_session_input and user_session_input in existing_sessions:
        session_id = user_session_input
        is_resuming = True
        print(f"\n[+] Resuming existing session: '{session_id}'")
    else:
        if user_session_input:
            session_id = user_session_input
            print(f"\n[+] Created new session with custom ID: '{session_id}'")
        else:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"\n[+] Created new session ID: '{session_id}'")

    config = {"configurable": {"thread_id": session_id}}

    # 2. Check existing messages in state
    current_state = graph.get_state(config)
    existing_messages = current_state.values.get("messages", []) if current_state else []

    # 3. Handle System Message for NEW sessions without triggering an early LLM call
    if not is_resuming and not existing_messages:
        user_sys_prompt = input("Enter System Prompt (or press ENTER for default): ").strip()
        sys_text = user_sys_prompt or "You are a helpful technical assistant."
        
        # Save SystemMessage to state directly using update_state (prevents early LLM output)
        graph.update_state(config, {"messages": [SystemMessage(content=sys_text)]})
        print("[+] System prompt configured and saved to session state.")
    else:
        print("[+] Preserving existing session state and system prompt.")

    print("=" * 50)

    # Calculate turn count (exclude SystemMessage from count if present)
    user_ai_messages = [m for m in existing_messages if not isinstance(m, SystemMessage)]
    turn = (len(user_ai_messages) // 2) + 1

    if existing_messages:
        print(f"\nLoaded {len(existing_messages)} total state messages from history.\n")

    # 4. Interactive chat loop
    while True:
        try:
            user_input = input(f"[Turn {turn}] You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                break

            graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            turn += 1

        except (KeyboardInterrupt, EOFError):
            break

    print("\nGoodbye!")

if __name__ == '__main__':
    main()