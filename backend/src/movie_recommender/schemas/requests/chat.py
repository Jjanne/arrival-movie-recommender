from typing import List, Optional

from pydantic import BaseModel


class ChatHistoryItem(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatHistoryItem]] = None


class ChatResponse(BaseModel):
    reply: str
