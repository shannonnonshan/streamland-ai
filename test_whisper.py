"""
StreamLand AI - Test Model
Test pipeline:
Whisper -> Transcript -> Summarization
"""

import os
import sys

from dotenv import load_dotenv

from utils.model_loader import ModelLoader
from models.summarization.interface import (
    SummarizationModel
)

load_dotenv()

# =========================================================
# TRANSCRIPTION
# =========================================================

def test_transcribe(
    audio_file,
    model
):

    print(
        f"\n[TEST] Transcribing: "
        f"{audio_file}"
    )

    try:

        result = model.transcribe(
            audio_file
        )

        transcript = (
            result.get("text", "")
            if isinstance(result, dict)
            else str(result)
        )

        language = (
            result.get("language", "unknown")
            if isinstance(result, dict)
            else "unknown"
        )

        print(
            f"\n[LANGUAGE] {language}"
        )

        print(
            f"\n[TEXT]\n"
        )

        print(transcript)

        print_timed_transcript(
            result
        )

        return result

    except Exception as e:

        print(
            f"[ERROR] "
            f"Transcription failed: {e}\n"
        )

        return None

# =========================================================
# EXTRACT TEXT
# =========================================================

def extract_transcript_text(
    transcript_result
):

    if transcript_result is None:
        return ""

    if isinstance(
        transcript_result,
        str
    ):
        return transcript_result

    if isinstance(
        transcript_result,
        dict
    ):

        text = transcript_result.get(
            "text"
        )

        return (
            text
            if isinstance(text, str)
            else ""
        )

    return str(
        transcript_result
    )

# =========================================================
# PRINT TIMESTAMPS
# =========================================================

def print_timed_transcript(
    transcript_result
):

    if not isinstance(
        transcript_result,
        dict
    ):
        return

    timestamps = (
        transcript_result.get(
            "timestamps"
        )
        or []
    )

    if not timestamps:
        return

    print("\n[TIMESTAMPS]")

    for item in timestamps:

        start = float(
            item.get(
                "start",
                0.0
            )
        )

        text = item.get(
            "text",
            ""
        )

        print(
            f"- {start:.2f}s: {text}"
        )

# =========================================================
# SUMMARIZATION
# =========================================================

def test_summarize(
    text,
    summarizer
):

    print(
        f"\n[TEST] Summarizing..."
    )

    print(
        f"[INFO] Input length: "
        f"{len(text)} chars"
    )

    try:

        if (
            not text
            or not text.strip()
        ):

            print(
                "[ERROR] Empty text"
            )

            return None

        # =================================================
        # LIMIT INPUT
        # =================================================

        MAX_INPUT_CHARS = 2000

        text = text[
            :MAX_INPUT_CHARS
        ]

        # =================================================
        # LANGUAGE DETECT
        # =================================================

        detected_lang = (
            summarizer.detect_language(
                text
            )
        )

        print(
            f"[LANGUAGE] "
            f"Detected: {detected_lang}"
        )

        if detected_lang is None:

            print(
                "[ERROR] "
                "Unsupported language"
            )

            return None

        # =================================================
        # SUMMARIZE
        # =================================================

        print(
            f"[MODEL] "
            f"Switching to: "
            f"{detected_lang}"
        )

        result = summarizer.infer(
            text
        )

        print(
            f"[MODEL] "
            f"Using model: "
            f"{summarizer.model_path}\n"
        )

        if result is False:

            print(
                "[ERROR] "
                "Summarization failed"
            )

            return None

        summary = result.get(
            "summary",
            ""
        )

        if not summary:

            print(
                "[ERROR] "
                "No summary generated"
            )

            return None

        print("\n[SUMMARY]")
        print("-" * 60)

        print(summary)

        print("-" * 60)

        print(
            f"[INFO] Length: "
            f"{len(summary)} chars\n"
        )

        return summary

    except Exception as e:

        print(
            f"[ERROR] "
            f"Summarization failed: {e}\n"
        )

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

        print(
            "[INFO] "
            "No MODEL_TYPE specified."
        )

        print(
            "[INFO] "
            "Using default Whisper model."
        )

        model_path = os.getenv(
            "WHISPER_MODEL_PATH",
            "shannonnonshan/streamland-whisper-ct2"
        )

        use_hf = (
            os.getenv(
                "WHISPER_USE_HF",
                "true"
            ).lower()
            == "true"
        )

        print(
            f"[INFO] Loading model "
            f"from "
            f"{'HF Hub' if use_hf else 'Local'}:"
        )

        print(model_path)

        try:

            model = (
                ModelLoader.load_model(
                    "whisper",
                    model_path=model_path,
                    from_hf=use_hf
                )
            )

            print(
                "\n[SUCCESS] "
                "Whisper loaded!"
            )

        except Exception as e:

            print(
                f"\n[ERROR] "
                f"Whisper load failed: {e}"
            )

            sys.exit(1)

    else:

        print(
            f"[SUCCESS] "
            f"Model loaded: "
            f"{model.info()}\n"
        )

    # =====================================================
    # LOAD SUMMARIZER
    # =====================================================

    summarizer = None

    # =====================================================
    # AUDIO FILES
    # =====================================================

    audio_files_env = os.getenv(
        "TEST_AUDIO_FILES",
        ""
    ).strip()

    if audio_files_env:

        audio_files = [
            item.strip()
            for item in (
                audio_files_env.split(",")
            )
            if item.strip()
        ]

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

        if not os.path.exists(
            audio_file
        ):

            print(
                f"[SKIP] "
                f"File not found: "
                f"{audio_file}"
            )

            continue

        print(
            f"[FILE] {audio_file}"
        )

        # =================================================
        # STEP 1 - TRANSCRIBE
        # =================================================

        transcript_result = (
            test_transcribe(
                audio_file,
                model
            )
        )

        transcript = (
            extract_transcript_text(
                transcript_result
            )
        )

        if not transcript.strip():

            print(
                "[SKIP] "
                "Empty transcript"
            )

            continue

        # =================================================
        # STEP 2 - LOAD SUMMARIZER
        # =================================================

        if summarizer is None:

            print(
                "\n[STEP 2] "
                "Loading summarizer..."
            )

            try:

                summarizer = (
                    SummarizationModel()
                )

                print(
                    "[SUCCESS] "
                    "Summarizer loaded!"
                )

            except Exception as e:

                print(
                    f"[ERROR] "
                    f"Summarizer load failed: {e}"
                )

                continue

        # =================================================
        # STEP 3 - SUMMARIZE
        # =================================================

        test_summarize(
            transcript,
            summarizer
        )

    print("\n===================================")
    print("ALL TESTS DONE")
    print("===================================\n")