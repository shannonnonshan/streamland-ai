"""FastAPI dependencies for fetching loaded model instances."""

from functools import lru_cache

from fastapi import HTTPException

from api.model_registry import models
from models.moderation.interface import ModerationModel


def _require_model(model_key: str, expected_type: str):
    model = models.get(model_key)
    if model is None:
        raise HTTPException(status_code=503, detail=f"{expected_type} model is not loaded")

    if getattr(model, "model_type", None) != expected_type:
        raise HTTPException(status_code=500, detail=f"Loaded model type mismatch: expected {expected_type}")

    return model


def get_whisper_model():
    return _require_model("whisper", "whisper")


def get_summarization_model():
    return _require_model("summarization", "summarization")


@lru_cache(maxsize=1)
def _get_moderation_singleton():
    return ModerationModel()


def get_moderation_model():
    return _get_moderation_singleton()
