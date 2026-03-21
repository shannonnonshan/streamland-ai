# Whisper Fine-tuned Model

Fine-tuned OpenAI Whisper model for speech-to-text transcription.

## Usage

```python
from models.whisper import WhisperModel

# Load from local path
model = WhisperModel(model_path="models/whisper/model/whisper-finetuned")

# Load from Hugging Face Hub
model = WhisperModel(model_path="username/whisper-finetuned", from_hf=True)

# Transcribe audio
transcript = model.transcribe("path/to/audio.wav")
```

## Configuration

- **Base Model**: openai/whisper-base
- **Language**: English (multilingual capable)
- **Audio Format**: WAV, MP3, FLAC, OGG
- **Sample Rate**: 16kHz (auto-resampled)
- **Chunk Length**: 30 seconds

## Installation

```bash
pip install torch transformers librosa
```

## License

Proprietary - StreamLand AI

