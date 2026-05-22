"""
StreamLand AI - API Server
Supports:
- Whisper GPU service
- Moderation/Summarization CPU service

SERVICE_MODE:
- gpu  -> Whisper routes only
- cpu  -> Moderation + Summarization routes
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
    try:
        model_registry.models.clear()

        # Load Whisper (STT) unless Replicate proxying is enabled
        if ModelConfig.REPLICATE_USE:
            print("[INIT] Replicate proxy enabled. Skipping local Whisper load.")
        else:
            print("[INIT] Loading Whisper model...")
            model_registry.models["whisper"] = ModelLoader.load_model(
                "whisper",
                model_path=ModelConfig.WHISPER_MODEL,
                from_hf=ModelConfig.WHISPER_USE_HF
            )
            print("✓ Whisper loaded")

        print("[INIT] Loading Embeddings model...")
        model_registry.models["embeddings"] = ModelLoader.load_model(
            "embeddings",
            model_path=ModelConfig.EMBEDDINGS_MODEL,
            from_hf=ModelConfig.EMBEDDINGS_USE_HF
        )
        print("✓ Embeddings loaded")

        print("[INIT] Loading Chatbot model...")
        model_registry.models["chatbot"] = ModelLoader.load_model(
            "chatbot",
            model_path=ModelConfig.CHATBOT_MODEL,
            from_hf=ModelConfig.CHATBOT_USE_HF
        )
        print("✓ Chatbot loaded")

        # Load Summarization
        print("[INIT] Loading Summarization model...")
        model_registry.models["summarization"] = ModelLoader.load_model(
            "summarization",
            model_path=ModelConfig.SUMMARIZATION_MODEL,
            from_hf=ModelConfig.SUMMARIZATION_USE_HF
        )
        print("✓ Summarization loaded")
        
        print("\n✓ All models loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load models: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize models on server startup."""
    init_models()


# Register routers
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
        "loaded": {key: model.info() for key, model in model_registry.models.items()}
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