import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from api.v1.context_assembler import ContextAssembler
from api.v1.db import (
    db_append_chat_turn,
    db_get_chat_history,
    db_get_profile,
    db_get_memories,
    db_add_memory,
)
from api.v1.engine import get_graph, make_thread_id, summarize_trimmed_turns

assembler = ContextAssembler(max_total_tokens=4096, reserve_output_tokens=512)


def get_profile(person_id: str) -> Dict[str, Any]:
    return db_get_profile(person_id) or {}


def get_relevant_memories(person_id: str, query: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    return db_get_memories(person_id, query=query, limit=limit)


def build_context(
    person_id: str,
    session_id: Optional[str],
    system_prompt: Optional[str],
    user_query: str,
    retrieved_docs: Optional[List[Dict[str, Any]]] = None,
    memories: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    profile = get_profile(person_id)
    history = db_get_chat_history(person_id, session_id or "")

    memory_items = memories if memories is not None else get_relevant_memories(person_id, query=user_query)

    return {
        "person_id": person_id,
        "session_id": session_id or "",
        "profile": profile,
        "history": history,
        "memories": memory_items,
        "retrieved_docs": retrieved_docs or [],
        "user_query": user_query,
        "system_prompt": system_prompt or "You are a helpful assistant.",
    }


def build_prompt(context: Dict[str, Any]) -> str:
    profile = context.get("profile") or {}
    history = context.get("history") or []
    memories = context.get("memories") or []
    retrieved_docs = context.get("retrieved_docs") or []
    user_query = context.get("user_query", "")
    system_prompt = context.get("system_prompt", "You are a helpful assistant.")

    profile_block = json.dumps(profile, indent=2, ensure_ascii=False) if profile else "No profile available."
    history_block = "\n".join(
        [f"{item.get('role', 'unknown').upper()}: {item.get('content', '')}" for item in history]
    ) or "No conversation history."
    memory_block = "\n".join(
        [f"- {item.get('content', '')}" for item in memories]
    ) or "No relevant memories."
    docs_block = "\n".join(
        [f"- {doc.get('text', '')}" for doc in retrieved_docs]
    ) or "No retrieved documents."

    return (
        f"{system_prompt}\n\n"
        f"PROFILE:\n{profile_block}\n\n"
        f"RELEVANT MEMORIES:\n{memory_block}\n\n"
        f"CONVERSATION HISTORY:\n{history_block}\n\n"
        f"RETRIEVED DOCS:\n{docs_block}\n\n"
        f"USER QUERY:\n{user_query}"
    )


def store_conversation_turn(person_id: str, session_id: str, user_query: str, ai_response: str):
    db_append_chat_turn(person_id, session_id, user_query, ai_response)
    return {"status": "stored"}


def process_chat_turn(
    person_id: str,
    session_id: str,
    system_prompt: str,
    user_query: str,
    retrieved_docs: list
) -> tuple[str, dict]:
    """Handles context assembly with running summary and executes graph turns."""
    thread_id = make_thread_id(person_id, session_id)
    config = {"configurable": {"thread_id": thread_id}}

    graph = get_graph()

    current_state = graph.get_state(config)
    raw_history = current_state.values.get("messages", []) if current_state else []
    existing_summary = current_state.values.get("summary", "") if current_state else ""

    history_dicts = []
    for m in raw_history:
        if isinstance(m, HumanMessage):
            history_dicts.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            history_dicts.append({"role": "assistant", "content": m.content})

    context = build_context(
        person_id=person_id,
        session_id=session_id,
        system_prompt=system_prompt,
        user_query=user_query,
        retrieved_docs=retrieved_docs or [],
        memories=get_relevant_memories(person_id, query=user_query),
    )

    prompt_text = build_prompt(context)

    assembled = assembler.assemble(
        system_prompt=prompt_text,
        user_query=user_query,
        history=history_dicts,
        retrieved_docs=retrieved_docs or [],
        summary=existing_summary
    )

    trimmed_turns = assembled.get("trimmed_history", [])
    updated_summary = existing_summary
    if trimmed_turns:
        updated_summary = summarize_trimmed_turns(existing_summary, trimmed_turns)
        if updated_summary != existing_summary:
            assembled = assembler.assemble(
                system_prompt=prompt_text,
                user_query=user_query,
                history=history_dicts,
                retrieved_docs=retrieved_docs or [],
                summary=updated_summary
            )

    assembled_messages = []
    for msg in assembled["messages"]:
        if msg["role"] == "system":
            assembled_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            assembled_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            assembled_messages.append(AIMessage(content=msg["content"]))

    result = graph.invoke({"messages": assembled_messages, "summary": updated_summary}, config=config)
    latest_msg = result["messages"][-1].content

    telemetry = assembled["telemetry"]
    telemetry["running_summary"] = updated_summary if updated_summary else "(None)"

    store_conversation_turn(person_id, session_id, user_query, latest_msg)

    return latest_msg, telemetry


def invoke_context_engineering(
    person_id: str,
    session_id: str,
    system_prompt: Optional[str],
    user_query: str,
    retrieved_docs: Optional[list] = None,
) -> Dict[str, Any]:
    profile = get_profile(person_id)
    history = db_get_chat_history(person_id, session_id)
    memories = get_relevant_memories(person_id, query=user_query)

    context = build_context(
        person_id=person_id,
        session_id=session_id,
        system_prompt=system_prompt,
        user_query=user_query,
        retrieved_docs=retrieved_docs or [],
        memories=memories,
    )

    prompt = build_prompt(context)

    response_text, telemetry = process_chat_turn(
        person_id=person_id,
        session_id=session_id,
        system_prompt=system_prompt or "You are a helpful assistant.",
        user_query=user_query,
        retrieved_docs=retrieved_docs or [],
    )

    return {
        "person_id": person_id,
        "session_id": session_id,
        "profile": profile,
        "history": history,
        "memories": memories,
        "context": context,
        "prompt": prompt,
        "response": response_text,
        "telemetry": telemetry,
    }