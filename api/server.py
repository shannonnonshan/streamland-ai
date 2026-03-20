import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify

load_dotenv()

app = Flask(__name__)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    from run import test_whisper
    import tempfile

    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio = request.files["file"]
    original_ext = os.path.splitext(audio.filename)[1] if audio.filename else ".mp3"
    with tempfile.NamedTemporaryFile(suffix=original_ext, delete=False) as tmp:
        audio.save(tmp.name)
        tmp_path = tmp.name

    try:
        transcript = test_whisper(tmp_path)
    except Exception as exc:
        return jsonify({"error": f"Transcription failed: {exc}"}), 500
    finally:
        os.remove(tmp_path)

    return jsonify({"transcript": transcript})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
