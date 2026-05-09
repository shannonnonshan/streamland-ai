"""Run Streamland Whisper on Replicate from a local audio file.

Usage:
  python run_replicate.py /path/to/audio.wav --language en

Requires `REPLICATE_API_TOKEN` in environment or .env file.
"""
import os
import sys
import json
import argparse
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run streamland-whisper on Replicate")
    parser.add_argument("audio", help="Path to local audio file")
    parser.add_argument("--language", "-l", default="", help="Language code (en, vi). Optional")
    parser.add_argument("--model", "-m", default="shannonnonshan/streamland-whisper", help="Replicate model (owner/model)")
    args = parser.parse_args()

    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        print("ERROR: REPLICATE_API_TOKEN not set in environment. Set it and retry.")
        sys.exit(1)

    try:
        import replicate
    except Exception as e:
        print("ERROR: replicate SDK not installed. Install with: pip install replicate")
        print(e)
        sys.exit(1)

    model_id = args.model
    # ensure version suffix
    if ":" not in model_id:
        model_id = model_id + ":latest"

    input_payload = {"audio": args.audio}
    if args.language:
        input_payload["language"] = args.language

    print(f"Running model on Replicate: {model_id}")
    print(f"Audio: {args.audio}")

    try:
        output = replicate.run(model_id, input=input_payload)
        print("\n=== Replicate Output ===")
        try:
            print(json.dumps(output, indent=2, ensure_ascii=False))
        except Exception:
            print(output)

    except Exception as e:
        print("ERROR: Replicate run failed:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
