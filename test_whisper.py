"""
StreamLand AI - Test Model
Test pipeline:
Whisper -> Transcript -> Summarization
"""

import os
import sys
import tempfile

import httpx
from dotenv import load_dotenv

from utils.model_loader import ModelLoader
from models.summarization.interface import SummarizationModel

load_dotenv()

# =========================================================
# DOWNLOAD AUDIO FROM URL
# =========================================================

def download_audio_from_url(url: str) -> str:
    """Download audio from URL to temp file, return temp file path."""
    print(f"\n[DOWNLOAD] Fetching audio from URL: {url}")
    try:
        with httpx.Client(timeout=300) as client:
            response = client.get(url)
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
            filename = url.split("?")[0].split("/")[-1] or "audio.wav"
            _, ext = os.path.splitext(filename)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".wav")
            tmp.write(response.content)
            tmp.close()
            print(f"[DOWNLOAD] Saved to: {tmp.name} ({len(response.content)} bytes)")
            return tmp.name
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        sys.exit(1)

# =========================================================
# TRANSCRIPTION
# =========================================================

def test_transcribe(audio_file, model):
    print(f"\n[TEST] Transcribing: {audio_file}")
    try:
        result = model.transcribe(audio_file)
        transcript = result.get("text", "") if isinstance(result, dict) else str(result)
        language = result.get("language", "unknown") if isinstance(result, dict) else "unknown"
        print(f"\n[LANGUAGE] {language}")
        print(f"\n[TEXT]\n")
        print(transcript)
        print_timed_transcript(result)
        return result
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}\n")
        return None

# =========================================================
# EXTRACT TEXT
# =========================================================

def extract_transcript_text(transcript_result):
    if transcript_result is None:
        return ""
    if isinstance(transcript_result, str):
        return transcript_result
    if isinstance(transcript_result, dict):
        text = transcript_result.get("text")
        return text if isinstance(text, str) else ""
    return str(transcript_result)

# =========================================================
# PRINT TIMESTAMPS
# =========================================================

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

# =========================================================
# SUMMARIZATION
# =========================================================

def test_summarize(text, summarizer):
    print(f"\n[TEST] Summarizing...")
    print(f"[INFO] Input length: {len(text)} chars")
    try:
        if not text or not text.strip():
            print("[ERROR] Empty text")
            return None

        MAX_INPUT_CHARS = 2000
        text = text[:MAX_INPUT_CHARS]

        detected_lang = summarizer.detect_language(text)
        print(f"[LANGUAGE] Detected: {detected_lang}")

        if detected_lang is None:
            print("[ERROR] Unsupported language")
            return None

        print(f"[MODEL] Switching to: {detected_lang}")
        result = summarizer.infer(text)
        print(f"[MODEL] Using model: {summarizer.model_path}\n")

        if result is False:
            print("[ERROR] Summarization failed")
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

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("\n===================================")
    print("STREAMLAND AI TEST")
    print("===================================\n")

    # =====================================================
    # LOAD WHISPER
    # =====================================================

    model = ModelLoader.from_env()

    if model is None:
        print("[INFO] No MODEL_TYPE specified.")
        print("[INFO] Using default Whisper model.")

        model_path = os.getenv("WHISPER_MODEL_PATH", "shannonnonshan/streamland-whisper-ct2")
        use_hf = os.getenv("WHISPER_USE_HF", "true").lower() == "true"

        print(f"[INFO] Loading model from {'HF Hub' if use_hf else 'Local'}:")
        print(model_path)

        try:
            model = ModelLoader.load_model("whisper", model_path=model_path, from_hf=use_hf)
            print("\n[SUCCESS] Whisper loaded!")
        except Exception as e:
            print(f"\n[ERROR] Whisper load failed: {e}")
            sys.exit(1)
    else:
        print(f"[SUCCESS] Model loaded: {model.info()}\n")

    # =====================================================
    # LOAD SUMMARIZER
    # =====================================================

    summarizer = None

    # =====================================================
    # AUDIO FILES - URL takes priority over local files
    # =====================================================

    audio_url = os.getenv(
        "TEST_AUDIO_URL",
        "https://pub-6ec835ecee45466fa5552dedffaee2e4.r2.dev/audio-export/75753de3-b366-4ba5-88ae-32fca5862ba1.wav"
    )

    if audio_url:
        tmp_path = download_audio_from_url(audio_url)
        audio_files = [tmp_path]
        _downloaded_tmp = tmp_path
    else:
        _downloaded_tmp = None
        audio_files_env = os.getenv("TEST_AUDIO_FILES", "").strip()
        if audio_files_env:
            audio_files = [item.strip() for item in audio_files_env.split(",") if item.strip()]
        else:
            audio_files = [
                "utils/data/audio/testaudio.mp3",
                "utils/data/audio/testaudio-vn.mp3",
            ]

    # =====================================================
    # PROCESS FILES
    # =====================================================

    for audio_file in audio_files:
        print("\n===================================")

        if not os.path.exists(audio_file):
            print(f"[SKIP] File not found: {audio_file}")
            continue

        print(f"[FILE] {audio_file}")

        transcript_result = test_transcribe(audio_file, model)
        transcript = extract_transcript_text(transcript_result)

        if not transcript.strip():
            print("[SKIP] Empty transcript")
            continue

        if summarizer is None:
            print("\n[STEP 2] Loading summarizer...")
            try:
                summarizer = SummarizationModel()
                print("[SUCCESS] Summarizer loaded!")
            except Exception as e:
                print(f"[ERROR] Summarizer load failed: {e}")
                continue

        test_summarize(transcript, summarizer)

    # =====================================================
    # CLEANUP DOWNLOADED TEMP FILE
    # =====================================================

    if _downloaded_tmp and os.path.exists(_downloaded_tmp):
        os.remove(_downloaded_tmp)
        print(f"\n[CLEANUP] Removed temp file: {_downloaded_tmp}")

    print("\n===================================")
    print("ALL TESTS DONE")
    print("===================================\n")