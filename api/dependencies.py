"""FastAPI dependencies for fetching loaded model instances."""

from fastapi import HTTPException

from api.model_registry import models


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


def get_embeddings_model():
    return _require_model("embeddings", "embeddings")


def get_chatbot_model():
    return _require_model("chatbot", "chatbot")
