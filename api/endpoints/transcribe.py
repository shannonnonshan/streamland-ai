"""Transcribe endpoint using Whisper"""

from fastapi import APIRouter, UploadFile, File, HTTPException
import os

router = APIRouter(prefix="/transcribe", tags=["transcribe"])


@router.post("")
async def transcribe(file: UploadFile = File(...), model=None):
    """Transcribe audio file using Whisper model."""
    if not model or model.model_type != "whisper":
        raise HTTPException(status_code=400, detail="Whisper model not available")
    
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        result = model.transcribe(temp_path)
        os.remove(temp_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
