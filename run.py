"""
StreamLand AI - Test Hugging Face Model
Purpose: Test script for Whisper model loaded from Hugging Face Hub
Inputs: Test audio files from utils/data/audio/
Output: Transcription results
"""

import os
import sys
from dotenv import load_dotenv
from models.whisper.interface import WhisperModel

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
    # Load model from Hugging Face Hub
    model_path = os.getenv("WHISPER_MODEL_PATH", "shannonnonshan/streamland-whisper")
    use_hf = os.getenv("WHISPER_USE_HF", "true").lower() == "true"
    
    print(f"[INFO] Loading model from {'HF Hub' if use_hf else 'Local'}: {model_path}")
    
    try:
        model = WhisperModel(model_path=model_path, from_hf=use_hf)
        print("[SUCCESS] Model loaded!\n")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)
    
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