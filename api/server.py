"""
StreamLand AI - API Server
Purpose: RESTful API for speech-to-text transcription using Whisper model
Supports loading model from local disk or Hugging Face Hub
Endpoints: /health, /transcribe
"""

import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import librosa
import numpy as np
from models.whisper.interface import WhisperModel

load_dotenv()

app = FastAPI(
    title="StreamLand AI API",
    description="Speech-to-Text Processing with Whisper",
    version="0.1.0"
)

# Initialize model
model = None

def init_model():
    """Initialize the model on startup."""
    global model
    try:
        model_path = os.getenv("WHISPER_MODEL_PATH", "shannonnonshan/streamland-whisper")
        use_hf = os.getenv("WHISPER_USE_HF", "true").lower() == "true"
        model = WhisperModel(model_path=model_path, from_hf=use_hf)
        print(f"✓ Model loaded: {model_path}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize model on server startup."""
    init_model()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_status = "ready" if model else "failed"
    return {
        "status": "ok",
        "model": model_status,
        "version": "0.1.0"
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe audio file using Whisper model."""
    if not model:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Save temp file
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Load and process audio
        audio, sr = librosa.load(temp_path, sr=16000)
        
        # Transcribe
        result = model.transcribe(audio)
        
        # Cleanup
        os.remove(temp_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "transcript": result["text"],
            "language": result.get("language", "unknown")
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("API_DEBUG", "false").lower() == "true"
    
    uvicorn.run(app, host=host, port=port, reload=debug)
