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
    turn_input: List[BaseMessage]
    summary: str
    
    
class ProfileUpsertRequest(BaseModel):
    profile_data: Dict[str, Any] = Field(default_factory=dict)


class MemoryCreateRequest(BaseModel):
    content: str
    memory_type: str = "general"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    expires_in_days: Optional[int] = Field(default=None, ge=1, description="TTL in days; omit for no expiry")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


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
    

class UserProfile(BaseModel):
    """User preferences, account details, and support background."""
    content: str
    preferred_name: Optional[str] = None
    prime_member: Optional[bool] = None
    preferences: List[str] = []


class CustomerTicketFact(BaseModel):
    """Key facts about orders, refunds, delays, and product issues."""
    content: str
    order_id: Optional[str] = None
    issue_type: Optional[str] = None
    status: Optional[str] = None