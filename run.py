"""
StreamLand AI - Full Test Pipeline
Whisper -> Clean -> RAG -> mT5 Summarization
"""

import os
import sys
import re
from dotenv import load_dotenv

from utils.model_loader import ModelLoader
from models.summarization.interface import SummarizationModel
from utils.rag_context import get_summarization_rag

load_dotenv()


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, max_chars=1500):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def test_transcribe(audio_file, model):
    print(f"\n[TEST] Transcribing: {audio_file}")

    try:
        result = model.transcribe(audio_file)

        if isinstance(result, dict):
            text = result.get("text", "")
        else:
            text = str(result)

        text = clean_text(text)

        print("\n[TRANSCRIPT]")
        print("-" * 60)
        print(text)
        print("-" * 60)
        print(f"[INFO] Length: {len(text)} chars\n")

        return text

    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}\n")
        return None


def test_summarize(text, summarizer, rag=None):
    print(f"\n[TEST] Summarizing...")
    print(f"[INFO] Input length: {len(text)} chars")

    try:
        if not text:
            print("[ERROR] Empty text, skipping summarization")
            return None

        chunks = chunk_text(text) if len(text) > 1500 else [text]
        print(f"[INFO] Chunks: {len(chunks)}")

        summaries = []

        for i, chunk in enumerate(chunks):
            print(f"[INFO] Chunk {i + 1}/{len(chunks)}")

            if rag:
                prompt = rag.build_prompt(chunk)
            else:
                prompt = f"Summarize the following text:\n{chunk}"

            summary = summarizer.summarize(
                prompt,
                max_length=120,
                min_length=30
            )

            if summary:
                summaries.append(summary)

        if not summaries:
            print("[ERROR] No summary generated")
            return None

        final_summary = " ".join(summaries)

        print("\n[SUMMARY]")
        print("-" * 60)
        print(final_summary)
        print("-" * 60)
        print(f"[INFO] Length: {len(final_summary)} chars\n")

        return final_summary

    except Exception as e:
        print(f"[ERROR] Summarization failed: {e}\n")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("[START] Pipeline")
    print("=" * 60)

    print("\n[INIT] RAG")
    rag = get_summarization_rag()

    print("\n[STEP 1] Load Whisper")
    model = ModelLoader.from_env()

    if model is None:
        model_path = os.getenv("WHISPER_MODEL_PATH", "openai/whisper-base")
        use_hf = os.getenv("WHISPER_USE_HF", "true").lower() == "true"

        try:
            model = ModelLoader.load_model(
                "whisper",
                model_path=model_path,
                from_hf=use_hf
            )
            print("[OK] Whisper loaded\n")
        except Exception as e:
            print(f"[ERROR] Whisper load failed: {e}")
            sys.exit(1)
    else:
        print("[OK] Whisper loaded\n")

    print("[STEP 2] Load mT5")
    mt5_path = os.getenv("MT5_MODEL_PATH") or os.getenv("FLAN_MODEL_PATH") or "google/mt5-small"

    try:
        summarizer = SummarizationModel(model_path=mt5_path)
        print("[OK] mT5 loaded\n")
    except Exception as e:
        print(f"[ERROR] mT5 load failed: {e}")
        sys.exit(1)

    print("[STEP 3] Run")
    print("-" * 60)

    audio_files = [
        "utils/data/audio/testaudio.mp3",
        "utils/data/audio/testaudio-vn.mp3"
    ]

    for audio_file in audio_files:
        if not os.path.exists(audio_file):
            print(f"[SKIP] Missing file: {audio_file}")
            continue

        text = test_transcribe(audio_file, model)

        # 🔥 FIX: chỉ cần có text là summarize luôn
        if text and len(text.strip()) > 0:
            test_summarize(text, summarizer, rag=rag)
        else:
            print("[SKIP] Empty transcription")

    print("-" * 60)
    print("[END]")
    print("=" * 60)