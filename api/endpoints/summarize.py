"""Summarization endpoint using mT5"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from utils.rag_context import get_summarization_rag

router = APIRouter(prefix="/summarize", tags=["summarize"])


class SummarizeRequest(BaseModel):
    """Request body for summarization."""
    text: str
    max_length: int = 150
    min_length: int = 30
    use_rag: bool = False
    rag_context_type: str = "general"  # streamland_ai, technical, general


class SummarizeResponse(BaseModel):
    """Response body for summarization."""
    status: str
    original_length: int
    summary: str
    summary_length: int
    compression_ratio: float


# Dependency to get summarization model from server
def get_model(request=None):
    """Get summarization model - to be set by server."""
    return None


@router.post("", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest, model=None):
    """
    Summarize text using mT5 model.
    
    Args:
        request: SummarizeRequest with text and optional parameters
        model: mT5 model instance
        
    Returns:
        SummarizeResponse with summary and statistics
    """
    if not model or model.model_type != "summarization":
        raise HTTPException(status_code=400, detail="Summarization model not available")
    
    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text must be at least 10 characters")
    
    try:
        # Augment text with RAG context if requested
        input_text = request.text
        if request.use_rag:
            rag = get_summarization_rag()
            input_text = rag.augment_for_summarization(
                request.text,
                context_type=request.rag_context_type
            )
        
        # Generate summary
        summary = model.summarize(
            input_text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        if not summary:
            raise HTTPException(status_code=500, detail="Failed to generate summary")
        
        # Calculate statistics
        compression_ratio = len(summary) / len(request.text)
        
        return SummarizeResponse(
            status="success",
            original_length=len(request.text),
            summary=summary,
            summary_length=len(summary),
            compression_ratio=round(compression_ratio, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@router.get("/info")
async def summarize_info(model=None):
    """Get summarization model information."""
    if not model or model.model_type != "summarization":
        raise HTTPException(status_code=400, detail="Summarization model not available")
    
    return {
        "model_type": "summarization",
        "model_path": model.model_path,
        "device": str(model.device),
        "max_length": 150,
        "min_length": 30
    }
