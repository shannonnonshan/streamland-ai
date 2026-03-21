"""Content moderation endpoint"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import os

router = APIRouter(prefix="/moderation", tags=["moderation"])


class TextModerationRequest(BaseModel):
    text: str


@router.post("/text")
async def moderate_text(request: TextModerationRequest, model=None):
    """Check if text content is safe."""
    if not model or model.model_type != "moderation":
        raise HTTPException(status_code=400, detail="Moderation model not available")
    
    try:
        result = model.moderate_text(request.text)
        return {
            "status": "success",
            "text": request.text,
            "moderation": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image")
async def moderate_image(file: UploadFile = File(...), model=None):
    """Check if image is NSFW safe."""
    if not model or model.model_type != "moderation":
        raise HTTPException(status_code=400, detail="Moderation model not available")
    
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        result = model.moderate_image(temp_path)
        os.remove(temp_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "moderation": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
