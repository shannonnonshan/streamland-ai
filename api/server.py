"""
API Server
Purpose: RESTful API server for StreamLand AI speech-to-text service
Exposes endpoints for audio transcription and model management.
Supports both local and Hugging Face Hub models.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pipelines import SpeechPipeline

# Load environment variables
load_dotenv()

app = FastAPI(
    title="StreamLand AI API",
    description="Speech-to-Text Processing API with Whisper",
    version="0.1.0"
)

# Initialize pipeline with configuration
def get_pipeline():
    """Get or create the speech pipeline with current config."""
    model_path = os.getenv("MODEL_PATH", "models/whisper/model/whisper-finetuned")
    use_hf = os.getenv("MODEL_USE_HF", "false").lower() == "true"
    
    pipeline = SpeechPipeline(model_path=model_path, use_hf=use_hf)
    return pipeline


# Global pipeline instance
try:
    pipeline = get_pipeline()
except Exception as e:
    print(f"⚠️  Warning: Failed to initialize pipeline: {e}")
    pipeline = None


@app.get("/health")
def health_check():
    """Health check endpoint."""
    model_status = "ready" if pipeline else "failed"
    return {
        "status": "ok",
        "model": model_status,
        "version": "0.1.0"
    }


@app.get("/config")
def get_config():
    """Get current configuration."""
    return {
        "model_path": os.getenv("MODEL_PATH", "models/whisper/model/whisper-finetuned"),
        "use_huggingface": os.getenv("MODEL_USE_HF", "false"),
        "device": os.getenv("DEVICE", "auto"),
        "sample_rate": int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
        "chunk_length": int(os.getenv("AUDIO_CHUNK_LENGTH", "30"))
    }


@app.post("/transcribe")
def transcribe(audio_file: str):
    """
    Transcribe an audio file.
    
    Args:
        audio_file (str): Path to audio file
        
    Returns:
        dict: Transcription result or error
    """
    if not pipeline:
        raise HTTPException(
            status_code=500,
            detail="Model not initialized. Check server logs."
        )
    
    if not os.path.exists(audio_file):
        raise HTTPException(
            status_code=400,
            detail=f"Audio file not found: {audio_file}"
        )
    
    try:
        text = pipeline.process_audio(audio_file)
        return {
            "transcript": text,
            "status": "success",
            "file": audio_file
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription error: {str(e)}"
        )


@app.post("/reload-model")
def reload_model():
    """Reload model from config (useful after config changes)."""
    global pipeline
    try:
        pipeline = get_pipeline()
        return {
            "status": "success",
            "message": "Model reloaded successfully"
        }
    except Exception as e:
        print(f"[ERROR] Failed to reload model: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": str(e)
            }
        )


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("API_DEBUG", "false").lower() == "true"
    
    print(f"[INFO] Starting API server at {host}:{port}")
    uvicorn.run(app, host=host, port=port, debug=debug)
