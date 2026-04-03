"""Pipeline endpoint combining Transcription + Summarization"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from utils.rag_context import get_summarization_rag

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


class PipelineResponse(BaseModel):
    """Response body for pipeline (transcribe + summarize)."""
    status: str
    filename: str
    transcription: str
    transcription_length: int
    summary: str
    summary_length: int
    compression_ratio: float


@router.post("/transcribe-summarize", response_model=PipelineResponse)
async def transcribe_summarize(
    file: UploadFile = File(...),
    max_summary_length: int = 150,
    min_summary_length: int = 30,
    use_rag: bool = False,
    rag_context_type: str = "general",
    models: dict = None
):
    """
    Transcribe audio + summarize text in one pipeline.
    
    Args:
        file: Audio file to transcribe
        max_summary_length: Max length for summary
        min_summary_length: Min length for summary
        use_rag: Whether to use RAG context for summarization
        rag_context_type: Type of RAG context (streamland_ai, technical, general)
        models: Dictionary of loaded models {
            "whisper": whisper_model,
            "summarization": summarization_model
        }
        
    Returns:
        PipelineResponse with transcription and summary
    """
    if not models or "whisper" not in models or "summarization" not in models:
        raise HTTPException(
            status_code=400,
            detail="Both Whisper and Summarization models required"
        )
    
    whisper_model = models["whisper"]
    summarization_model = models["summarization"]
    
    if whisper_model.model_type != "whisper":
        raise HTTPException(status_code=400, detail="Invalid Whisper model")
    
    if summarization_model.model_type != "summarization":
        raise HTTPException(status_code=400, detail="Invalid Summarization model")
    
    temp_path = None
    try:
        # Step 1: Transcribe audio
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        transcription = whisper_model.transcribe(temp_path)
        
        if not transcription or len(str(transcription).strip()) < 10:
            raise HTTPException(status_code=400, detail="No valid transcription generated")
        
        # Step 2: Summarize transcription
        input_text = transcription
        if use_rag:
            rag = get_summarization_rag()
            input_text = rag.augment_for_summarization(
                transcription,
                context_type=rag_context_type
            )
        
        summary = summarization_model.summarize(
            input_text,
            max_length=max_summary_length,
            min_length=min_summary_length
        )
        
        if not summary:
            raise HTTPException(status_code=500, detail="Failed to generate summary")
        
        # Calculate statistics
        compression_ratio = len(summary) / len(transcription)
        
        return PipelineResponse(
            status="success",
            filename=file.filename,
            transcription=transcription,
            transcription_length=len(transcription),
            summary=summary,
            summary_length=len(summary),
            compression_ratio=round(compression_ratio, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(e)}"
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/info")
async def pipeline_info(models: dict = None):
    """Get pipeline (transcription + summarization) info."""
    if not models or "whisper" not in models or "summarization" not in models:
        raise HTTPException(
            status_code=400,
            detail="Both models required"
        )
    
    return {
        "pipeline": "transcribe + summarize",
        "whisper": {
            "model_type": "whisper",
            "model_path": models["whisper"].model_path,
            "device": str(models["whisper"].device)
        },
        "summarization": {
            "model_type": "summarization",
            "model_path": models["summarization"].model_path,
            "device": str(models["summarization"].device),
            "max_length": 150,
            "min_length": 30
        }
    }
