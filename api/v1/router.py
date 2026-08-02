from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

from api.v1.schema import ChatRequest, ChatResponse
from api.v1.db import db_get_all_threads, db_delete_thread, db_get_chat_history
from api.v1.engine import get_graph, make_thread_id
from api.v1.services import process_chat_turn

router = APIRouter(tags=["Context Chat & State Management"])


@router.post("/chat/invoke", response_model=ChatResponse, summary="Invoke context-budgeted multi-turn chat")
def invoke_chat(req: ChatRequest):
    response_text, telemetry = process_chat_turn(
        person_id=req.person_id,
        session_id=req.session_id,
        system_prompt=req.system_prompt,
        user_query=req.user_query,
        retrieved_docs=req.retrieved_docs
    )

    return ChatResponse(
        person_id=req.person_id,
        session_id=req.session_id,
        response=response_text,
        telemetry=telemetry
    )

@router.get("/persons", summary="Fetch list of unique person IDs with saved history")
def get_persons():
    threads = db_get_all_threads()
    persons = list(set(t.split("::")[0] for t in threads if "::" in t))
    return {"persons": persons}

@router.get("/sessions/{person_id}", summary="Fetch active session IDs for a person_id")
def get_sessions(person_id: str):
    threads = db_get_all_threads()
    sessions = [t.split("::")[1] for t in threads if t.startswith(f"{person_id}::")]
    return {"person_id": person_id, "sessions": sessions}

# @router.get("/chat/history/{person_id}/{session_id}", summary="Retrieve conversation history")
# def get_chat_history(person_id: str, session_id: str):
#     thread_id = make_thread_id(person_id, session_id)
#     config = {"configurable": {"thread_id": thread_id}}
    
#     current_state = graph.get_state(config)
#     messages = current_state.values.get("messages", []) if current_state else []
    
#     formatted_history = []
#     for msg in messages:
#         if isinstance(msg, HumanMessage):
#             formatted_history.append({"role": "user", "content": msg.content})
#         elif isinstance(msg, AIMessage):
#             formatted_history.append({"role": "assistant", "content": msg.content})
            
#     return {"person_id": person_id, "session_id": session_id, "history": formatted_history}

@router.get("/chat/history/{person_id}/{session_id}")
def get_chat_history(person_id: str, session_id: str):
    messages = db_get_chat_history(person_id, session_id)
    thread_id = make_thread_id(person_id, session_id)
    config = {"configurable": {"thread_id": thread_id}}

    graph = get_graph()
    state = None
    try:
        state = graph.get_state(config)
    except Exception:
        state = None

    if not state or not getattr(state, "values", None):
        formatted_messages = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in messages
        ]
        try:
            graph.update_state(config, {"messages": formatted_messages})
        except Exception:
            pass

    return {"person_id": person_id, "session_id": session_id, "history": messages}


@router.delete("/session/{person_id}/{session_id}", summary="Delete a specific session")
def delete_session(person_id: str, session_id: str):
    thread_id = make_thread_id(person_id, session_id)
    deleted_count = db_delete_thread(thread_id)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session ID not found.")
    return {"status": "deleted", "deleted_rows": deleted_count}

@router.delete("/person/{person_id}", summary="Delete all records for a person_id")
def delete_person(person_id: str):
    deleted_count = db_delete_thread(f"{person_id}::")
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Person ID not found.")
    return {"status": "deleted", "deleted_rows": deleted_count}