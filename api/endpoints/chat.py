"""Chat endpoint using Llama RAG"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    context: str = None


@router.post("")
async def chat(request: ChatRequest, model=None):
    """Chat with video content context using Llama."""
    if not model or model.model_type != "llama":
        raise HTTPException(status_code=400, detail="LLM model not available")
    
    try:
        prompt = f"Context: {request.context}\n\nQuestion: {request.message}" if request.context else request.message
        response = model.generate(prompt)
        
        return {
            "status": "success",
            "message": request.message,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
