# StreamLand AI - Speech-to-Text Processing

Production-ready speech recognition system with fine-tuned OpenAI Whisper model.

## Quick Start

### Initial Setup

First time setup - configure Hugging Face credentials:

```bash
python setup.py
```

Or manually create `.env` from `.env.example`:

```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN from https://huggingface.co/settings/tokens
```

### Installation

```bash
pip install -r requirements.txt
```

### Local Usage

```python
from models.whisper import WhisperModel

model = WhisperModel(model_path="models/whisper/model/whisper-finetuned")
transcript = model.transcribe("path/to/audio.wav")
print(transcript)
```

### Load from Hugging Face

```python
from models.whisper import WhisperModel

model = WhisperModel(
    model_path="username/whisper-finetuned",
    from_hf=True
)
transcript = model.transcribe("path/to/audio.wav")
```

## Project Structure

```
streamland-ai/
├── run.py                          # Entry point & CLI
├── requirements.txt                # Dependencies
├── api/
│   └── server.py                   # FastAPI server
├── models/whisper/
│   ├── interface.py                # WhisperModel class
│   └── model/whisper-finetuned/    # Model weights & config
├── pipelines/
│   └── speech_pipeline.py          # Processing pipeline
└── utils/
    └── huggingface_utils.py        # HF utilities
```

## Push Model to Hugging Face

Configure credentials in `.env`:

```bash
# Automatic setup
python setup.py

# Or manual:
cp .env.example .env
# Edit .env: add HF_TOKEN and HF_REPO_ID
```

Get token from: https://huggingface.co/settings/tokens (create with 'write' permission)

Push model:

```bash
python push_to_huggingface.py username/whisper-finetuned
```

## API Server

```bash
python -m uvicorn api.server:app
```

**Endpoints:**
- `GET /health` - Health check
- `GET /config` - Server configuration
- `POST /transcribe?audio_file=path` - Transcribe audio
- `POST /reload-model` - Reload model

## Command Line

```bash
python run.py utils/data/audio/sample.wav
python run.py utils/data/audio/sample.wav --hf  # Use HF model
```

## Configuration

Via `.env` file or environment variables:

```bash
MODEL_PATH=models/whisper/model/whisper-finetuned
MODEL_USE_HF=false
API_HOST=0.0.0.0
API_PORT=8000
HF_TOKEN=hf_xxxxx
```

See `.env.example` for all options.

## Supported Audio Formats

WAV, MP3, FLAC, OGG (via librosa)

## Hardware Requirements

- GPU: CUDA-enabled GPU (4GB+ VRAM recommended)
- CPU: Supported but slower

## License

Proprietary - StreamLand AI

