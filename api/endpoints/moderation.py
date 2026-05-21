"""Content moderation endpoint."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_moderation_model

router = APIRouter(prefix="/moderation", tags=["moderation"])


class TextModerationRequest(BaseModel):
    text: str = Field(..., min_length=1)


def _extract_toxic_words(moderation: dict) -> list[str]:
    toxic_words = []
    for span in moderation.get("matched_spans", []):
        text = span.get("text")
        if text and text not in toxic_words:
            toxic_words.append(text)
    return toxic_words


@router.post("/text")
async def moderate_text(request: TextModerationRequest, model=Depends(get_moderation_model)):
    """Run the staged moderation flow on text input."""
    try:
        moderation = model.moderate_text(request.text, rewrite=False)
        return {
            "status": "success",
            "text": request.text,
            "moderation": {
                "status": moderation.get("label", "SAFE"),
                "toxic_word": _extract_toxic_words(moderation),
                "label": moderation.get("label", "SAFE"),
                "score": moderation.get("score", 0.0),
                "categories": moderation.get("categories", []),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/image")
async def moderate_image():
    """Image moderation is not part of the staged text flow."""
    raise HTTPException(status_code=501, detail="Image moderation is not implemented in this flow")
