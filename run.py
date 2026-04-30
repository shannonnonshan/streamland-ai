"""
StreamLand AI - Test Model
Test pipeline: Whisper → Transcript → BART/T5 Summarization
"""

import os
import sys
from dotenv import load_dotenv

from utils.model_loader import ModelLoader
from utils.config import ModelConfig
from models.summarization.interface import SummarizationModel

load_dotenv()


# -------------------------
# Transcription
# -------------------------
def test_transcribe(audio_file, model):
    print(f"\n[TEST] Transcribing: {audio_file}")

    try:
        result = model.transcribe(audio_file)
        transcript = result.get("text", "") if isinstance(result, dict) else str(result)
        print(f"[TEXT] {transcript}\n")
        print_timed_transcript(result)
        return result

    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}\n")
        return None


def extract_transcript_text(transcript_result):
    if transcript_result is None:
        return ""

    if isinstance(transcript_result, str):
        return transcript_result

    if isinstance(transcript_result, dict):
        text = transcript_result.get("text")
        return text if isinstance(text, str) else ""

    return str(transcript_result)


def print_timed_transcript(transcript_result):
    if not isinstance(transcript_result, dict):
        return

    timestamps = transcript_result.get("timestamps") or []
    if not timestamps:
        return

    print("\n[TIMESTAMPS]")
    for item in timestamps:
        start = float(item.get("start", 0.0))
        text = item.get("text", "")
        print(f"- {start:.2f}s: {text}")


# -------------------------
# Summarization
# -------------------------
def test_summarize(text, summarizer):
    print(f"\n[TEST] Summarizing...")
    print(f"[INFO] Input length: {len(text)} chars")

    try:
        if not text or not text.strip():
            print("[ERROR] Empty text, skipping summarization")
            return None

        # Trim input to avoid seq2seq context overflow.
        MAX_INPUT_CHARS = 2000
        text = text[:MAX_INPUT_CHARS]

        prompt = text

        # DEBUG: Detect language
        detected_lang = summarizer.detect_language(prompt)
        print(f"[LANGUAGE] Detected: {detected_lang}")

        if detected_lang is None:
            print("[ERROR] Unsupported language, returning False")
            return None

        # Use new infer method with language detection
        print(f"[MODEL] Switching to language: {detected_lang}")
        result = summarizer.infer(prompt)
        print(f"[MODEL] Now using: {summarizer.model_path}\n")

        if result is False:
            print("[ERROR] Unsupported language or summarization failed")
            return None

        summary = result.get("summary", "")

        if not summary:
            print("[ERROR] No summary generated")
            return None

        print("\n[SUMMARY]")
        print("-" * 60)
        print(summary)
        print("-" * 60)
        print(f"[INFO] Length: {len(summary)} chars\n")

        return summary

    except Exception as e:
        print(f"[ERROR] Summarization failed: {e}\n")
        return None


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # -------------------------
    # Load speech model (Whisper or from env)
    # -------------------------
    model = ModelLoader.from_env()

    if model is None:
        print("[INFO] No MODEL_TYPE specified. Using default Whisper model.")

        model_path = os.getenv(
            "WHISPER_MODEL_PATH",
            "shannonnonshan/streamland-whisper"
        )
        use_hf = os.getenv("WHISPER_USE_HF", "true").lower() == "true"

        print(f"[INFO] Loading model from {'HF Hub' if use_hf else 'Local'}: {model_path}")

        try:
            model = ModelLoader.load_model(
                "whisper",
                model_path=model_path,
                from_hf=use_hf
            )
            print("[SUCCESS] Whisper model loaded!\n")

        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            sys.exit(1)

    else:
        print(f"[SUCCESS] Model loaded: {model.info()}\n")

    # -------------------------
    # Load summarizer (lazy load)
    # -------------------------
    summarizer = None

    # -------------------------
    # Test files
    # -------------------------
    audio_files_env = os.getenv("TEST_AUDIO_FILES", "").strip()
    if audio_files_env:
        audio_files = [item.strip() for item in audio_files_env.split(",") if item.strip()]
    else:
        audio_files = [
            "utils/data/audio/testaudio.mp3",
            "utils/data/audio/testaudio-vn.mp3"
        ]

    for audio_file in audio_files:
        if not os.path.exists(audio_file):
            print(f"[SKIP] File not found: {audio_file}")
            continue

        # STEP 1: Transcribe
        transcript_result = test_transcribe(audio_file, model)
        transcript = extract_transcript_text(transcript_result)

        if not transcript.strip():
            print("[SKIP] Empty transcript\n")
            continue

        # STEP 2: Load summarizer once
        if summarizer is None:
            print("[STEP 2] Loading summarization model...")

            try:
                summarizer = SummarizationModel()
                print("[SUCCESS] Summarization model loaded!\n")

            except Exception as e:
                print(f"[ERROR] Summarization model load failed: {e}")
                continue

        # STEP 3: Summarize
        test_summarize(transcript, summarizer)