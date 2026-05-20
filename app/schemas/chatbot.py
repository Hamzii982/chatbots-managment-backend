from pydantic import BaseModel, Field
from typing import Optional

class ChatbotBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Chatbot name")
    system_prompt: str = Field(..., min_length=10, max_length=12000, description="System prompt for the chatbot")
    model_id: int = Field(..., gt=0, description="Model ID (must be positive)")
    retriever_type: str = Field(default="mock", min_length=1, max_length=50, description="Retriever type")
    chunk_size: int = Field(default=500, ge=100, le=5000, description="Chunk size for documents")
    chunk_overlap: int = Field(default=50, ge=0, le=1000, description="Overlap between chunks")
    reranker_type: str = Field(default="simple", min_length=1, max_length=50, description="Reranker type")
    top_k: int = Field(default=3, ge=1, le=50, description="Number of top results to return")
    short_term_memory: bool = Field(default=True, description="Enable short-term memory")
    long_term_memory: bool = Field(default=False, description="Enable long-term memory")
    is_active: bool = Field(default=True, description="Is chatbot active")

class ChatbotCreate(ChatbotBase):
    pass

class ChatbotResponse(ChatbotBase):
    id: int
    is_vectorized: bool

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    thread_id: Optional[int] = None
    disable_memory: bool = False


class ChatResponse(BaseModel):
    response: str
    sources: list[str] | None = None

    class Config:
        from_attributes = True