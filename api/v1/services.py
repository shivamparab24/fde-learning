from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from api.v1.db import db_append_chat_turn
from api.v1.engine import get_graph, make_thread_id, summarize_trimmed_turns
from api.v1.context_assembler import ContextAssembler

assembler = ContextAssembler(max_total_tokens=4096, reserve_output_tokens=512)

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

    # 1. Read existing checkpoint state
    current_state = graph.get_state(config)
    raw_history = current_state.values.get("messages", []) if current_state else []
    existing_summary = current_state.values.get("summary", "") if current_state else ""

    # 2. Extract plain text turn history
    history_dicts = []
    for m in raw_history:
        if isinstance(m, HumanMessage):
            history_dicts.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            history_dicts.append({"role": "assistant", "content": m.content})

    # 3. Assemble budgeting strategy including running summary
    assembled = assembler.assemble(
        system_prompt=system_prompt,
        user_query=user_query,
        history=history_dicts,
        retrieved_docs=retrieved_docs or [],
        summary=existing_summary
    )

    # 4. Check if turns were trimmed; update running summary if needed
    trimmed_turns = assembled.get("trimmed_history", [])
    updated_summary = existing_summary
    if trimmed_turns:
        updated_summary = summarize_trimmed_turns(existing_summary, trimmed_turns)
        # Re-assemble if summary changed to update payload system message cleanly
        if updated_summary != existing_summary:
            assembled = assembler.assemble(
                system_prompt=system_prompt,
                user_query=user_query,
                history=history_dicts,
                retrieved_docs=retrieved_docs or [],
                summary=updated_summary
            )

    # 5. Map back to LangChain message structures
    assembled_messages = []
    for msg in assembled["messages"]:
        if msg["role"] == "system":
            assembled_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            assembled_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            assembled_messages.append(AIMessage(content=msg["content"]))

    # 6. Invoke state graph & save checkpoint with summary
    result = graph.invoke({"messages": assembled_messages, "summary": updated_summary}, config=config)
    latest_msg = result["messages"][-1].content

    # Add active summary text preview to telemetry output
    telemetry = assembled["telemetry"]
    telemetry["running_summary"] = updated_summary if updated_summary else "(None)"

    # Persist simple chat record for UI/history
    db_append_chat_turn(person_id, session_id, user_query, latest_msg)

    return latest_msg, telemetry