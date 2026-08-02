from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    person_id: str
    session_id: str
    system_prompt: Optional[str]
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