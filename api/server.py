"""
StreamLand AI - API Server
Supports:
- Whisper GPU service
- Moderation/Summarization CPU service
"""

import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api import model_registry
from api.endpoints import transcribe, chat, summarize, moderation
from api.endpoints import search as search_endpoint
from utils.config import ModelConfig
from utils.model_loader import ModelLoader
from api.endpoints.admin import router as admin_router

# =========================
# Load env
# =========================

load_dotenv()

SERVICE_MODE = os.getenv("SERVICE_MODE", "cpu")

# =========================
# Optional GPU probe
# =========================

try:
    import torch
except Exception:
    torch = None

# =========================
# Logging
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger("streamland.api")

# =========================
# FastAPI app
# =========================

app = FastAPI(
    title="StreamLand AI API",
    description="AI inference API",
    version="1.0.0"
)

app.include_router(admin_router)

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Request logging
# =========================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()

    logger.info(
        "Request started: %s %s",
        request.method,
        request.url.path
    )

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Request finished: %s %s status=%s duration=%.2fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response


def init_models():
    """Initialize all models on startup."""
    model_registry.models.clear()

    load_plan = [
        ("whisper", ModelConfig.WHISPER_MODEL, ModelConfig.WHISPER_USE_HF),
        ("summarization", ModelConfig.SUMMARIZATION_MODEL, ModelConfig.SUMMARIZATION_USE_HF),
        ("moderation", ModelConfig.MODERATION_MODEL, ModelConfig.MODERATION_USE_HF),
        ("embeddings", ModelConfig.EMBEDDINGS_MODEL, ModelConfig.EMBEDDINGS_USE_HF),
        ("chatbot", ModelConfig.CHATBOT_MODEL, ModelConfig.CHATBOT_USE_HF),
    ]

    loaded_models = []
    failed_models = []

    for model_type, model_path, use_hf in load_plan:
        print(f"[INIT] Loading {model_type.capitalize()} model...")
        try:
            model_registry.models[model_type] = ModelLoader.load_model(
                model_type,
                model_path=model_path,
                from_hf=use_hf,
            )
            print(f"✓ {model_type.capitalize()} loaded")
            loaded_models.append(model_type)
        except Exception as exc:
            failed_models.append((model_type, str(exc)))
            print(f"✗ Failed to load {model_type}: {exc}")

    if failed_models:
        print("\n[INIT] Some models failed to load:")
        for model_type, error in failed_models:
            print(f"  - {model_type}: {error}")

    if loaded_models:
        print(f"\n✓ Loaded models: {', '.join(loaded_models)}")
    else:
        print("\n✗ No models loaded successfully!")


@app.on_event("startup")
async def startup_event():
    """Initialize models on server startup."""
    init_models()


# =========================
# Register routers
# =========================

app.include_router(transcribe.router)
app.include_router(search_endpoint.router)
app.include_router(chat.router)
app.include_router(moderation.router)
app.include_router(summarize.router)


@app.get("/health")
async def health_check():

    gpu_info = {
        "torch_present": bool(torch),
        "cuda_available": False,
        "device_count": 0,
        "devices": []
    }

    if torch:
        try:
            gpu_info["cuda_available"] = torch.cuda.is_available()
            gpu_info["device_count"] = torch.cuda.device_count()

            devices = []

            for i in range(gpu_info["device_count"]):
                try:
                    devices.append(torch.cuda.get_device_name(i))
                except Exception:
                    devices.append(f"cuda:{i}")

            gpu_info["devices"] = devices

        except Exception:
            pass

    return {
        "status": "ok",
        "service_mode": SERVICE_MODE,
        "gpu": gpu_info,
        "version": "1.0.0"
    }


@app.get("/models")
async def list_models():
    """List loaded models with their info."""

    return {
        "available": [
            {"name": "whisper", "type": "STT", "purpose": "Speech-to-Text"},
            {"name": "embeddings", "type": "Embeddings", "purpose": "Search & Recommendations"},
            {"name": "chatbot", "type": "Chat", "purpose": "Conversational QA"},
            {"name": "moderation", "type": "Safety", "purpose": "Content Moderation"},
            {"name": "summarization", "type": "NLG", "purpose": "Text Summarization"},
        ],
        "loaded": {
            key: model.info()
            for key, model in model_registry.models.items()
        }
    }


if __name__ == "__main__":

    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))

    logger.info(
        "Starting StreamLand AI API on %s:%s mode=%s",
        host,
        port,
        SERVICE_MODE
    )

    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )