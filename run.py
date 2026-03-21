"""
StreamLand AI - Test Model
Purpose: Test script for models loaded from Hugging Face Hub or local disk
Supports: Dynamic model selection via MODEL_TYPE and MODEL_PATH env vars
Inputs: Test audio files from utils/data/audio/
Output: Transcription results
"""

import os
import sys
from dotenv import load_dotenv
from utils.model_loader import ModelLoader

load_dotenv()


def test_transcribe(audio_file, model):
    """Transcribe audio file using loaded model."""
    print(f"\n[TEST] Transcribing: {audio_file}")
    
    try:
        result = model.transcribe(audio_file)
        print(f"[RESULT] {result}\n")
        return result
        
    except Exception as e:
        print(f"[ERROR] {e}\n")
        return None


if __name__ == "__main__":
    # Try to load model from MODEL_TYPE env var, fallback to Whisper
    model = ModelLoader.from_env()
    
    if model is None:
        # Fallback to Whisper if not configured
        print("[INFO] No MODEL_TYPE specified. Using default Whisper model.")
        model_path = os.getenv("WHISPER_MODEL_PATH", "shannonnonshan/streamland-whisper")
        use_hf = os.getenv("WHISPER_USE_HF", "true").lower() == "true"
        print(f"[INFO] Loading model from {'HF Hub' if use_hf else 'Local'}: {model_path}")
        
        try:
            model = ModelLoader.load_model("whisper", model_path=model_path, from_hf=use_hf)
            print("[SUCCESS] Model loaded!\n")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            sys.exit(1)
    else:
        print(f"[SUCCESS] Model loaded: {model.info()}\n")
    
    # Test with available audio files
    audio_files = [
        "utils/data/audio/testaudio.mp3",
        "utils/data/audio/testaudio-vn.mp3"
    ]
    
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            test_transcribe(audio_file, model)
        else:
            print(f"[SKIP] File not found: {audio_file}")