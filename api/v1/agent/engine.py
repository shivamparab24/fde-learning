import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from api.v1.agent.llm import get_llm
from api.v1.schema.schema import GraphState
from core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()
llm = get_llm()

_graph = None
_saver_cm: Optional[object] = None


def _build_graph(settings):
    saver_cm = PostgresSaver.from_conn_string(settings.database_url)
    saver = saver_cm.__enter__()
    saver.setup()

    def llm_node(state: GraphState):
        response = llm.invoke(state["turn_input"])
        return {"messages": [response]}

    builder = StateGraph(GraphState)
    builder.add_node("llm", llm_node)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)

    compiled_graph = builder.compile(checkpointer=saver)
    return compiled_graph, saver_cm


def initialize_graph():
    """Initialize the graph and DB connection. Call at app startup."""
    global _graph, _saver_cm
    if _graph is not None:
        return
    _graph, _saver_cm = _build_graph(settings)
    logger.info("LangGraph initialized with Postgres checkpointer.")


def close_graph():
    """Release the PostgresSaver context manager. Call at app shutdown."""
    global _graph, _saver_cm
    if _saver_cm is not None:
        try:
            _saver_cm.__exit__(None, None, None)
        except Exception as e:
            logger.warning("Error closing PostgresSaver: %s", e)
    _graph = None
    _saver_cm = None


def get_graph():
    if _graph is None:
        raise RuntimeError("Graph not initialized. Call initialize_graph() at app startup.")
    return _graph


def make_thread_id(person_id: str, session_id: str) -> str:
    return f"{person_id}::{session_id}"
