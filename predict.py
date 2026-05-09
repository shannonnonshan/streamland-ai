import cog
from faster_whisper import WhisperModel


class Predictor(cog.Predictor):

    def setup(self):

        self.model = WhisperModel(
            "shannonnonshan/streamland-whisper-ct2",
            device="cuda",
            compute_type="float16"
        )

        print("✓ Faster Whisper loaded")

    @cog.input(
        "audio",
        type=cog.Path,
        description="Audio file"
    )
    @cog.input(
        "language",
        type=str,
        default="",
        description="Language code like vi, en"
    )
    def predict(
        self,
        audio: cog.Path,
        language: str = ""
    ):

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