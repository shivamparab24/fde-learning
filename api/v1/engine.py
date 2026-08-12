from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage

from api.v1.llm import get_llm
from api.v1.db import _ensure_checkpoint_schema, _get_conn
from api.v1.schema import GraphState
from core.config import get_settings

settings = get_settings()
llm = get_llm()

_graph = None
_conn: Optional[object] = None


def _build_graph(settings):
    with _get_conn() as conn: 
        _ensure_checkpoint_schema(conn)
        
    # Enter the context manager manually to extract the raw PostgresSaver instance
    cm = PostgresSaver.from_conn_string(settings.database_url)
    saver = cm.__enter__()
    saver.setup()

    def llm_node(state: GraphState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(GraphState)
    builder.add_node("llm", llm_node)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)

    compiled_graph = builder.compile(checkpointer=saver)
    
    # Return graph and the context manager (so you can close it on app shutdown)
    return compiled_graph, cm


def summarize_trimmed_turns(existing_summary: str, trimmed_turns: list) -> str:
    """Summarizes older/trimmed conversation turns into a single updated summary string."""

    print(f"[Summary] generating rolling summary from {len(trimmed_turns)} trimmed turns")
    if not trimmed_turns:
        return existing_summary

    formatted_turns = "\n".join(
        [f"{msg['role'].upper()}: {msg['content']}" for msg in trimmed_turns]
    )

    if existing_summary.strip():
        prompt = (
            f"Existing summary of earlier conversation:\n{existing_summary}\n\n"
            f"New conversation turns to integrate:\n{formatted_turns}\n\n"
            "Write a single concise summary paragraph incorporating all key facts, topics, preferences, and decisions."
        )
    else:
        prompt = (
            f"Summarize key facts, topics, decisions, and user details from these conversation turns into a concise summary paragraph:\n{formatted_turns}"
        )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        summary_text = response.content.strip()
        print(f"[Summary] rolling summary generated: {len(summary_text)} chars")
        return summary_text
    except Exception as e:
        print(f"Error generating running summary: {e}")
        return existing_summary
    
  
def initialize_graph():
    """Initialize the graph and DB connection. Call at app startup."""
    global _graph, _conn
    if _graph is not None:
        return
    _graph, _conn = _build_graph(settings)


def get_graph():
    if _graph is None:
        raise RuntimeError("Graph not initialized. Call initialize_graph() at app startup.")
    return _graph


def make_thread_id(person_id: str, session_id: str) -> str:
    return f"{person_id}::{session_id}"