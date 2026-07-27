import time
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from core.config import get_settings

settings = get_settings()

# ------------------------------------------------------------------
# 1. Initialize ChatOllama with persistent settings
# ------------------------------------------------------------------
# Note: Keep static instructions at the very beginning of your prompt pipeline 
# to take full advantage of Ollama's automatic prefix caching.
llm = ChatOllama(
    base_url=settings.ollama_base_url,  # Your local or remote Ollama server
    model=settings.ollama_model_name,                  # Ensure you have pulled this model locally
    temperature=0.7,
    keep_alive="1h",                   # CRITICAL: Keeps KV cache in VRAM across turns
)

# A heavy, static system prompt (Prefix Caching Target)
HEAVY_SYSTEM_PROMPT = """
You are a highly specialized technical assistant.
Rules and Constraints:
- Provide concise, highly accurate code blocks.
- Never use external unverified libraries without explaining them.
- Always explain concepts using software engineering best practices.
- Maintain a warm and professional tone.
""" + "".join("\n- Supplemental knowledge context block line " + str(i) for i in range(100))
# ------------------------------------------------------------------
# 2. Define State Schema & Graph Nodes
# ------------------------------------------------------------------
class ChatState(TypedDict):
    # 'add_messages' appends new messages rather than overwriting existing history
    messages: Annotated[list[BaseMessage], add_messages]

def call_model(state: ChatState):
    # Structure messages so the static system prompt is ALWAYS the exact same prefix
    full_prompt = [SystemMessage(content="".join(HEAVY_SYSTEM_PROMPT))] + state["messages"]
    
    start_time = time.perf_counter()
    response = llm.invoke(full_prompt)
    latency = (time.perf_counter() - start_time) * 1000  # ms
    
    # Extract Ollama-specific prompt evaluation metrics
    eval_duration_ms = "N/A"
    if response.response_metadata and "prompt_eval_duration" in response.response_metadata:
        # Converts nanoseconds to milliseconds
        eval_duration_ms = round(response.response_metadata["prompt_eval_duration"] / 1e6, 2)

    print(f"\n[Metrics] Total Response Time: {latency:.2f} ms | Prompt Prefill Time: {eval_duration_ms} ms")
    return {"messages": [response]}

# ------------------------------------------------------------------
# 3. Build & Compile Graph with Checkpointer Persistence
# ------------------------------------------------------------------
builder = StateGraph(ChatState)
builder.add_node("assistant", call_model)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

# In-memory checkpointer retains thread history across multi-turn interactions
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# ------------------------------------------------------------------
# 4. Multi-turn Execution Demo
# ------------------------------------------------------------------
def run_chat_session():
    # Thread ID identifies the persistent user conversation session
    config = {"configurable": {"thread_id": "session_user_42"}}

    print("--- Turn 1: Initial Cold Request (Fills Prompt Cache) ---")
    input_1 = {"messages": [HumanMessage(content="Hello! Can you help me understand state persistence?")]}
    res_1 = graph.invoke(input_1, config=config)
    print("Assistant:", res_1["messages"][-1].content[:150], "...\n")

    print("--- Turn 2: Follow-up Request (Hits Local KV Cache) ---")
    input_2 = {"messages": [HumanMessage(content="What did I just ask you about?")]}
    res_2 = graph.invoke(input_2, config=config)
    print("Assistant:", res_2["messages"][-1].content[:150], "...\n")

if __name__ == "__main__":
    run_chat_session()