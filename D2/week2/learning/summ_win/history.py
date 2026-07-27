from typing import Annotated
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import trim_messages, SystemMessage, HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from core.config import get_settings
settings = get_settings()

llm = ChatOllama(
    model=settings.ollama_model_name,
    base_url=settings.ollama_base_url,
    temperature=0.0
)



# 1. WINDOWED HISTORY
class WindowState(TypedDict):
    messages: Annotated[list, add_messages]

def window_node(state: WindowState):
    trimmed = trim_messages(
        state["messages"],
        max_tokens=2,
        strategy="last",
        token_counter=len,
        start_on="human"
    )
    response = llm.invoke(trimmed)
    return {"messages": [response]}

builder_w = StateGraph(WindowState)
builder_w.add_node("chatbot", window_node)
builder_w.add_edge(START, "chatbot")
builder_w.add_edge("chatbot", END)
window_graph = builder_w.compile(checkpointer=MemorySaver())




# 2. RUNNING-SUMMARY HISTORY
class SummaryState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str

def summary_chat_node(state: SummaryState):
    summary = state.get("summary", "")
    msgs = state["messages"]
    
    if summary:
        full_context = [SystemMessage(content=f"Summary of past talk: {summary}")] + msgs
    else:
        full_context = msgs
        
    response = llm.invoke(full_context)
    return {"messages": [response]}

def summarize_node(state: SummaryState):
    summary = state.get("summary", "")
    msgs = state["messages"]
    
    prompt = f"Existing summary: '{summary}'. Add key points from new messages to update summary:" if summary else "Summarize the key user facts in 1 sentence:"
    res = llm.invoke(msgs + [HumanMessage(content=prompt)])
    
    delete_old = [RemoveMessage(id=m.id) for m in msgs[:-2]]
    return {"summary": res.content, "messages": delete_old}

def check_summary_needed(state: SummaryState):
    if len(state["messages"]) > 4:
        return "summarize"
    return END

builder_s = StateGraph(SummaryState)
builder_s.add_node("chatbot", summary_chat_node)
builder_s.add_node("summarize", summarize_node)
builder_s.add_edge(START, "chatbot")
builder_s.add_conditional_edges("chatbot", check_summary_needed)
builder_s.add_edge("summarize", END)
summary_graph = builder_s.compile(checkpointer=MemorySaver())

# 3. DEMO RUNNER
def run_demo():
    cfg = {"configurable": {"thread_id": "demo"}}
    turns = [
        "My favorite color is emerald green and I love baking bread.",
        "What is 10 + 10?",
        "What are my favorite things/hobbies that I mentioned earlier?"
    ]

    print("--- WINDOWED DEMO ---")
    for turn in turns:
        print(f"\nUser: {turn}")
        out = window_graph.invoke({"messages": [("user", turn)]}, config=cfg)
        print(f"Bot:  {out['messages'][-1].content}")

    print("\n" + "="*50 + "\n")

    print("--- RUNNING-SUMMARY DEMO ---")
    for turn in turns:
        print(f"\nUser: {turn}")
        out = summary_graph.invoke({"messages": [("user", turn)]}, config=cfg)
        print(f"Bot:  {out['messages'][-1].content}")

if __name__ == "__main__":
    run_demo()