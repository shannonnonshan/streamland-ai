"""
StreamLand AI - API Server
Purpose: Multi-model API for speech, text, embeddings, and content moderation
Supports 5 models: Whisper, Embeddings, Llama, Moderation, Summarization
Dynamic model loading from Hugging Face Hub or local disk
"""

import logging
import os
import time

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from utils.model_loader import ModelLoader
from utils.config import ModelConfig
from api.endpoints import transcribe, search, chat, recommend, moderation, summarize
from api import model_registry

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger("streamland.api")

app = FastAPI(
    title="StreamLand AI API",
    description="Multi-model AI API for speech, text, embeddings, and content moderation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    logger.info("Request started: %s %s", request.method, request.url.path)

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

        # Load Whisper (STT)
        print("[INIT] Loading Whisper model...")
        model_registry.models["whisper"] = ModelLoader.load_model(
            "whisper",
            model_path=ModelConfig.WHISPER_MODEL,
            from_hf=ModelConfig.WHISPER_USE_HF
        )
        print("✓ Whisper loaded")

        # Future models to enable later:
        # print("[INIT] Loading Embeddings model...")
        # model_registry.models["embeddings"] = ModelLoader.load_model(
        #     "embeddings",
        #     model_path=ModelConfig.EMBEDDINGS_MODEL,
        #     from_hf=ModelConfig.EMBEDDINGS_USE_HF
        # )
        # print("✓ Embeddings loaded")

        # print("[INIT] Loading LLama model...")
        # model_registry.models["llama"] = ModelLoader.load_model(
        #     "llama",
        #     model_path=ModelConfig.LLAMA_MODEL,
        #     from_hf=ModelConfig.LLAMA_USE_HF
        # )
        # print("✓ LLama loaded")

        # print("[INIT] Loading Moderation model...")
        # model_registry.models["moderation"] = ModelLoader.load_model(
        #     "moderation",
        #     model_path=ModelConfig.MODERATION_MODEL,
        #     from_hf=ModelConfig.MODERATION_USE_HF
        # )
        # print("✓ Moderation loaded")
        
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
# app.include_router(search.router)
# app.include_router(chat.router)
# app.include_router(recommend.router)
# app.include_router(moderation.router)
app.include_router(summarize.router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models_loaded": list(model_registry.models.keys()),
        "total_models": len(model_registry.models),
        "version": "1.0.0"
    }


@app.get("/models")
async def list_models():
    """List loaded models with their info."""
    return {
        "available": [
            {"name": "whisper", "type": "STT", "purpose": "Speech-to-Text"},
            # Future models to add later:
            # {"name": "embeddings", "type": "Embeddings", "purpose": "Search & Recommendations"},
            # {"name": "llama", "type": "LLM", "purpose": "Chat & RAG"},
            # {"name": "moderation", "type": "Safety", "purpose": "Content Moderation"},
            {"name": "summarization", "type": "NLG", "purpose": "Text Summarization"},
        ],
        "loaded": {key: model.info() for key, model in model_registry.models.items()}
    }


if __name__ == "__main__":
    import uvicorn
    from utils.config import ModelConfig
    
    print(f"\n🚀 Starting StreamLand AI API on {ModelConfig.API_HOST}:{ModelConfig.API_PORT}")
    uvicorn.run(app, host=ModelConfig.API_HOST, port=ModelConfig.API_PORT, reload=False, log_level="info")
