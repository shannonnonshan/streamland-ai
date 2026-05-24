"""Summarization endpoint."""

import logging

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, field_validator

from api.dependencies import get_summarization_model

router = APIRouter(prefix="/summarize", tags=["summarize"])
logger = logging.getLogger(__name__)


class SummarizeRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Text must not be empty")
        return v.strip()


@router.post("")
async def summarize(
    request: SummarizeRequest,
    model=Depends(get_summarization_model),
):
    """Summarize input text using the loaded summarization model."""
    text = request.text  # already stripped by validator

    logger.info("Summarize request received: input_length=%d", len(text))

    try:
        result = model.infer(text)

    except Exception:
        logger.exception("Summarize model.infer() crashed: input_length=%d", len(text))
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "data": None,
                "error": "Summarization model failed unexpectedly",
            },
        )

    # infer() returns False for unsupported/undetected language
    if result is False:
        logger.warning("Unsupported language: input_length=%d", len(text))
        raise HTTPException(
            status_code=422,
            detail={
                "status": "error",
                "data": None,
                "error": "Unsupported language — only English and Vietnamese are supported",
            },
        )

    # Guard against unexpected return types
    if not isinstance(result, dict):
        logger.error("Unexpected infer() return type: %s", type(result))
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "data": None,
                "error": "Summarization returned unexpected result format",
            },
        )

    summary = (result.get("summary") or "").strip()

    if not summary:
        logger.warning(
            "Summarization returned empty output: input_length=%d", len(text)
        )
        raise HTTPException(
            status_code=422,
            detail={
                "status": "error",
                "data": None,
                "error": "Summarization returned empty output — input may be too short",
            },
        )

    logger.info(
        "Summarize completed: input_length=%d summary_length=%d",
        len(text),
        len(summary),
    )

    return {
        "status": "success",
        "data": {
            "input_length": len(text),
            "summary": summary,
        },
        "error": None,
    }