from typing import Optional

from fastapi import APIRouter, HTTPException

from api.v1.db.db import (
    db_get_all_threads,
    db_get_chat_history,
    db_upsert_profile,
    db_add_memory,
    db_delete_session,
    db_delete_person,
)
from api.v1.memory.memory_service import LANGMEM_AVAILABLE, get_langmem_manager
from api.v1.schema.schema import (
    ChatRequest,
    ChatResponse,
    ContextBuildRequest,
    ConversationTurnStoreRequest,
    InvokeRequest,
    InvokeResponse,
    MemoryCreateRequest,
    ProfileUpsertRequest,
    PromptBuildRequest,
)
from api.v1.context.services import (
    invoke_context_engineering,
    process_chat_turn,
    get_profile,
    get_relevant_memories,
    build_context,
    build_prompt,
    store_conversation_turn,
)

router = APIRouter(tags=["Context Chat & State Management"])




@router.post("/invoke", response_model=InvokeResponse, summary="Single entrypoint for context-engineered chat")
def invoke_context_engineering_endpoint(req: InvokeRequest):
    result = invoke_context_engineering(
        person_id=req.person_id,
        session_id=req.session_id,
        system_prompt=req.system_prompt,
        user_query=req.user_query,
        retrieved_docs=req.retrieved_docs,
    )
    return result

    
    

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


@router.get("/profiles/{person_id}", summary="Get a customer profile")
def get_profile_endpoint(person_id: str):
    profile = get_profile(person_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return {"person_id": person_id, "profile": profile}


@router.post("/profiles/{person_id}", summary="Create or update a customer profile")
def upsert_profile_endpoint(person_id: str, payload: ProfileUpsertRequest):
    profile = db_upsert_profile(person_id, payload.profile_data)
    return {"person_id": person_id, "profile": profile}


@router.post("/memories/{person_id}", summary="Store a new memory for a person")
def create_memory_endpoint(person_id: str, payload: MemoryCreateRequest):
    if LANGMEM_AVAILABLE:
        langmem = get_langmem_manager()
        langmem.add_memory(
            person_id=person_id, 
            messages=[{"role": "user", "content": payload.content}], 
            metadata=payload.metadata
        )
    db_add_memory(person_id, payload.content, payload.memory_type, payload.metadata)
    return {"person_id": person_id, "status": "stored", "engine": "langmem" if LANGMEM_AVAILABLE else "postgres"}


@router.get("/memories/{person_id}", summary="Get relevant memories for a person")
def get_memories_endpoint(person_id: str, query: Optional[str] = None, limit: int = 5):
    memories = get_relevant_memories(person_id, query=query, limit=limit)
    return {"person_id": person_id, "query": query, "memories": memories}


@router.post("/context/build", summary="Build a reusable context object")
def build_context_endpoint(req: ContextBuildRequest):
    context = build_context(
        person_id=req.person_id,
        session_id=req.session_id,
        system_prompt=req.system_prompt,
        user_query=req.user_query,
        retrieved_docs=req.retrieved_docs or [],
        memories=req.memories,
    )
    return {"person_id": req.person_id, "session_id": req.session_id, "context": context}


@router.post("/prompt/build", summary="Build a prompt from context")
def build_prompt_endpoint(req: PromptBuildRequest):
    return {"prompt": build_prompt(req.context)}


@router.post("/conversation-turns", summary="Store a conversation turn explicitly")
def store_conversation_turn_endpoint(req: ConversationTurnStoreRequest):
    result = store_conversation_turn(req.person_id, req.session_id, req.user_query, req.ai_response)
    return result


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


# @router.get("/chat/history/{person_id}/{session_id}")
# def get_chat_history(person_id: str, session_id: str):
#     messages = db_get_chat_history(person_id, session_id)
#     thread_id = make_thread_id(person_id, session_id)
#     config = {"configurable": {"thread_id": thread_id}}

#     graph = get_graph()
#     state = None
#     try:
#         state = graph.get_state(config)
#     except Exception:
#         state = None

#     if not state or not getattr(state, "values", None):
#         formatted_messages = [
#             HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
#             for m in messages
#         ]
#         try:
#             graph.update_state(config, {"messages": formatted_messages})
#         except Exception:
#             pass

#     return {"person_id": person_id, "session_id": session_id, "history": messages}

@router.get("/chat/history/{person_id}/{session_id}")
def get_chat_history(person_id: str, session_id: str):
    messages = db_get_chat_history(person_id, session_id)
    return {"person_id": person_id, "session_id": session_id, "history": messages}


@router.delete("/session/{person_id}/{session_id}", summary="Delete a specific session")
def delete_session(person_id: str, session_id: str):
    deleted_count = db_delete_session(person_id, session_id)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session ID not found.")
    return {"status": "deleted", "deleted_rows": deleted_count}


@router.delete("/person/{person_id}", summary="Delete all records for a person_id")
def delete_person(person_id: str):
    deleted_count = db_delete_person(person_id)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Person ID not found.")
    return {"status": "deleted", "deleted_rows": deleted_count}
