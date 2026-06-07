from pydantic import BaseModel
from typing import Optional


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    sql: Optional[str] = None
    chart: Optional[dict] = None
    columns: list[str] = []
    rows: list[dict] = []
    warning: Optional[str] = None


class ChatMessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: int
    sql: Optional[str] = None
    chart: Optional[dict] = None
    columns: Optional[list[str]] = None
    rows: Optional[list[dict]] = None
    warning: Optional[str] = None

    model_config = {"from_attributes": True}


class ChatHistoryResponse(BaseModel):
    long_term_memory: Optional[str] = None
    messages: list[ChatMessageResponse] = []
