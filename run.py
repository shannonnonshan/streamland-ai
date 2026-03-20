"""
Main Entry Point
Purpose: Testing and demonstration script for the StreamLand AI system.
Used for quick testing of audio transcription functionality.
Supports both local and Hugging Face Hub models.
"""

import os
import sys
from dotenv import load_dotenv
from pipelines import SpeechPipeline


def test_audio_transcription(audio_file: str, use_hf: bool = False) -> str:
    """
    Test transcription on an audio file.
    
    Args:
        audio_file (str): Path to audio file to transcribe
        use_hf (bool): Load model from Hugging Face Hub
        
    Returns:
        str: Transcribed text
    """
    model_path = os.getenv(
        "MODEL_PATH",
        "models/whisper/model/whisper-finetuned"
    )
    
    pipeline = SpeechPipeline(model_path=model_path, use_hf=use_hf)
    result = pipeline.process_audio(audio_file)
    return result


def main():
    """Main entry point."""
    # Load environment variables from .env
    load_dotenv()
    
    # Example usage
    print("=" * 60)
    print("StreamLand AI - Speech-to-Text Demo")
    print("=" * 60)
    print()
    
    # Check if audio file is provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        use_hf = "--hf" in sys.argv or os.getenv("MODEL_USE_HF", "false").lower() == "true"
        
        if os.path.exists(audio_file):
            print(f"[INFO] Audio file: {audio_file}")
            print(f"[INFO] Model source: {'Hugging Face Hub' if use_hf else 'Local disk'}")
            print()
            print("[INFO] Transcribing...")
            
            try:
                transcript = test_audio_transcription(audio_file, use_hf=use_hf)
                print()
                print("[INFO] Transcription complete:")
                print("-" * 60)
                print(f"{transcript}")
                print("-" * 60)
            except Exception as e:
                print(f"[ERROR] {e}")
                sys.exit(1)
        else:
            print(f"[ERROR] Audio file not found: {audio_file}")
            sys.exit(1)
    else:
        # Print usage instructions
        print("Usage:")
        print()
        print("  python run.py <audio_file>")
        print("  python run.py <audio_file> --hf  (use Hugging Face model)")
        print()
        print("Environment variables:")
        print("  MODEL_PATH          - Local model path (default: models/whisper/model/whisper-finetuned)")
        print("  MODEL_USE_HF        - Use HF Hub if 'true' (default: false)")
        print("  HF_USERNAME         - HF username")
        print("  HF_REPO_BASE        - Repo base name (repos created as HF_USERNAME/HF_REPO_BASE-<model_name>)")
        print()
        print("Example:")
        print("  python run.py utils/data/audio/sample.wav")
        print("  python run.py utils/data/audio/sample.wav --hf")
        print()


if __name__ == "__main__":
    main()
