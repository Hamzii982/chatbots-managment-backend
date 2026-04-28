from pydantic import BaseModel, Field

class ModelBase(BaseModel):
    provider: str = Field(..., min_length=1, max_length=100, description="LLM provider")
    name: str = Field(..., min_length=1, max_length=255, description="Model configuration name")
    model_name: str = Field(..., min_length=1, max_length=255, description="Actual model name")
    api_key: str = Field(..., min_length=10, description="API key for the provider (WILL BE HASHED)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature parameter")
    max_tokens: int = Field(default=1000, ge=1, le=100000, description="Max tokens to generate")
    is_active: bool = Field(default=True, description="Is model active")

class ModelCreate(ModelBase):
    pass

class ModelResponse(BaseModel):
    id: int
    provider: str
    name: str
    model_name: str
    temperature: float
    max_tokens: int
    is_active: bool
    # NOTE: api_key intentionally excluded from response for security

    class Config:
        from_attributes = True