from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

from api.v1.context.prompt import DEFAULT_AMAZON_SYSTEM_PROMPT


class ChatRequest(BaseModel):
    person_id: str
    session_id: str
    system_prompt: Optional[str] = Field(default=DEFAULT_AMAZON_SYSTEM_PROMPT)
    user_query: str
    retrieved_docs: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    person_id: str
    session_id: str
    response: str
    telemetry: Dict[str, Any]


class GraphState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str
    
    
class ProfileUpsertRequest(BaseModel):
    profile_data: Dict[str, Any] = Field(default_factory=dict)


class MemoryCreateRequest(BaseModel):
    content: str
    memory_type: str = "general"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextBuildRequest(BaseModel):
    person_id: str
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None
    user_query: str
    retrieved_docs: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    memories: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


class PromptBuildRequest(BaseModel):
    context: Dict[str, Any]


class ConversationTurnStoreRequest(BaseModel):
    person_id: str
    session_id: str
    user_query: str
    ai_response: str


class InvokeRequest(BaseModel):
    person_id: str
    session_id: str
    system_prompt: Optional[str] = Field(default=DEFAULT_AMAZON_SYSTEM_PROMPT)
    user_query: str
    retrieved_docs: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


class InvokeResponse(BaseModel):
    person_id: str
    session_id: str
    profile: Dict[str, Any]
    history: List[Dict[str, str]]
    memories: List[Dict[str, Any]]
    context: Dict[str, Any]
    prompt: str
    response: str
    telemetry: Dict[str, Any]