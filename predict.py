from cog import BasePredictor, Input, Path
from faster_whisper import WhisperModel


class Predictor(BasePredictor):

    def setup(self):

        self.model = WhisperModel(
            "shannonnonshan/streamland-whisper-ct2",
            device="cuda",
            compute_type="float16"
        )

        print("✓ Model loaded")

    def predict(
        self,
        audio: Path = Input(description="Audio file"),
        language: str = Input(
            description="Language code",
            default=""
        )
    ) -> dict:

        segments, info = self.model.transcribe(
            str(audio),
            language=language if language else None
        )

        chunks = []
        full_text = ""

        for segment in segments:

            text = segment.text.strip()

            full_text += text + " "

            chunks.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": text
            })

        return {
            "text": full_text.strip(),
            "chunks": chunks,
            "language": info.language
        }