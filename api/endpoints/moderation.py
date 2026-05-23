"""Content moderation endpoint."""

import json
import logging
import traceback
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_moderation_model

router = APIRouter(prefix="/moderation", tags=["moderation"])
logger = logging.getLogger(__name__)


class TextModerationRequest(BaseModel):
    text: Any = Field(...)


def _preview_text(text: str, limit: int = 160) -> str:
    preview = " ".join((text or "").split())
    if len(preview) <= limit:
        return preview
    return f"{preview[:limit].rstrip()}..."


def _extract_plain_text(payload: Any) -> str:
    if payload is None:
        return ""

    if isinstance(payload, str):
        candidate = payload.strip()
        if not candidate:
            return ""

        if candidate[:1] in {"{", "["}:
            try:
                parsed = json.loads(candidate)
            except Exception:
                return candidate
            return _extract_plain_text(parsed)

        return candidate

    if isinstance(payload, dict):
        full_text = payload.get("full_text")
        if isinstance(full_text, str) and full_text.strip():
            return full_text.strip()

        segments = payload.get("segments")
        if isinstance(segments, list):
            parts = []
            for segment in segments:
                if isinstance(segment, dict):
                    segment_text = segment.get("text")
                    if isinstance(segment_text, str) and segment_text.strip():
                        parts.append(segment_text.strip())
            if parts:
                return " ".join(parts).strip()

        nested_text = payload.get("text")
        if nested_text is not None and nested_text is not payload:
            return _extract_plain_text(nested_text)

        return ""

    if isinstance(payload, list):
        parts = []
        for item in payload:
            item_text = _extract_plain_text(item)
            if item_text:
                parts.append(item_text)
        return " ".join(parts).strip()

    return ""


@router.post("/text")
async def moderate_text(
    request: TextModerationRequest,
    model=Depends(get_moderation_model),
):
    try:
        raw_text = request.text
        logger.info("Moderation raw transcript type=%s", type(raw_text).__name__)

        text = _extract_plain_text(raw_text)
        logger.info("Moderation normalized text preview=%s", _preview_text(text))
        logger.info("Moderation payload length=%s", len(text))

        if not text:
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "data": None,
                    "error": "Text must not be empty after normalization",
                },
            )

        moderation_raw = model.moderate_text(text)
        moderation = {
            "status": moderation_raw.get("status"),
            "toxic_word": moderation_raw.get("toxic_word") or [],
            "score": moderation_raw.get("score", 0.0),
            "categories": moderation_raw.get("categories") or [],
        }

        logger.info(
            "Moderation response preview=%s",
            {
                "status": moderation["status"],
                "toxic_word": moderation["toxic_word"],
                "score": moderation["score"],
                "categories": moderation["categories"],
            },
        )

        return {
            "status": "success",
            "moderation": moderation,
        }

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

@router.post("/image")
async def moderate_image():
    raise HTTPException(
        status_code=501,
        detail="Image moderation is not implemented in this flow",
    )