from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=128)
    max_tokens: int = Field(8, ge=1, le=128)

class GenerateResponse(BaseModel):
    text: str
 