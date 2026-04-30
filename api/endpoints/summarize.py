"""Summarization endpoint."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.dependencies import get_summarization_model

router = APIRouter(prefix="/summarize", tags=["summarize"])


class SummarizeRequest(BaseModel):
    text: str


@router.post("")
async def summarize(request: SummarizeRequest, model=Depends(get_summarization_model)):
    """Summarize input text using loaded summarization model."""
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "data": None,
                "error": "Text must not be empty",
            },
        )

    try:
        result = model.infer(text)
        if result is False:
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "data": None,
                    "error": "Unsupported language for summarization",
                },
            )

        summary = (result.get("summary") or "").strip()
        if not summary:
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "data": None,
                    "error": "Summarization returned empty output",
                },
            )

        return {
            "status": "success",
            "data": {
                "input_length": len(text),
                "summary": summary,
            },
            "error": None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "data": None,
                "error": str(e),
            },
        )
