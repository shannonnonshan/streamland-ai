"""Content moderation endpoint."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_moderation_model

router = APIRouter(prefix="/moderation", tags=["moderation"])


class TextModerationRequest(BaseModel):
    text: str = Field(..., min_length=1)
    rewrite: bool = True


@router.post("/text")
async def moderate_text(request: TextModerationRequest, model=Depends(get_moderation_model)):
    """Run the staged moderation flow on text input."""
    try:
        return {
            "status": "success",
            "text": request.text,
            "moderation": model.moderate_text(request.text, rewrite=request.rewrite),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/image")
async def moderate_image():
    """Image moderation is not part of the staged text flow."""
    raise HTTPException(status_code=501, detail="Image moderation is not implemented in this flow")
