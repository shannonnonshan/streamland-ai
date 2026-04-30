"""Transcribe endpoint using Whisper"""

import os
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

from api.dependencies import get_whisper_model

router = APIRouter(prefix="/transcribe", tags=["transcribe"])


@router.post("")
async def transcribe(file: UploadFile = File(...), model=Depends(get_whisper_model)):
    """Transcribe audio file using Whisper model."""
    temp_path = None

    try:
        _, ext = os.path.splitext(file.filename or "")
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".wav") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        result = model.transcribe(temp_path)

        return {
            "status": "success",
            "data": {
                "filename": file.filename,
                "result": result,
            },
            "error": None,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "data": None,
                "error": str(e),
            },
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
