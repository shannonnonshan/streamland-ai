# StreamLand AI - Speech-to-Text Service

Fine-tuned Whisper model for multilingual speech recognition.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure .env
HF_TOKEN=your_token
HF_USERNAME=your_username
WHISPER_MODEL_PATH=your_username/streamland-whisper
WHISPER_USE_HF=true

# Test model
python run.py

# Start API server
python -m api.server
```

## Project Structure

```
├── run.py                    # Test script
├── api/server.py             # FastAPI server
├── models/whisper/interface.py # Model wrapper
├── utils/model_pusher.py     # Push to Hugging Face
└── .env                      # Configuration
```

## Usage

**Test:** `python run.py`

**API Server:** `python -m api.server`

**Push Model:** `python utils/model_pusher.py whisper`

## API Endpoints

- `GET /health` - Health check
- `POST /transcribe` - Transcribe audio

## Configuration

Set these in `.env`:
- `HF_TOKEN` - Hugging Face token
- `HF_USERNAME` - Your username
- `WHISPER_MODEL_PATH` - Model path on HF Hub
- `API_PORT` - Server port (default: 8000)

## Features

- Fine-tuned Whisper model for English & Vietnamese
- Auto GPU/CPU detection
- Modular design for multiple models
- Universal model pusher for HF Hub
- FastAPI REST server
- Support for MP3, WAV, FLAC formats
