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

# =========================
# Root
# =========================

@app.get("/")
async def root():
    return {
        "ok": True,
        "service_mode": SERVICE_MODE,
        "message": "StreamLand AI API is running"
    }

# =========================
# Health
# =========================

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

# =========================
# Dynamic router loading
# =========================

if SERVICE_MODE == "gpu":

    logger.info("Loading GPU Whisper routes...")

    from api.endpoints import transcribe

    app.include_router(transcribe.router)

elif SERVICE_MODE == "cpu":

    logger.info("Loading CPU text routes...")

    from api.endpoints import moderation, summarize

    app.include_router(moderation.router)
    app.include_router(summarize.router)

else:

    logger.warning(
        "Unknown SERVICE_MODE=%s. No routes loaded.",
        SERVICE_MODE
    )

# =========================
# Entrypoint
# =========================

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