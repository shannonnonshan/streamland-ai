# StreamLand AI - Speech-to-Text Service

Fine-tuned Whisper model for multilingual speech recognition.

## Quick Start

```bash
## Install dependencies
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

## API Endpoints (Working)

Base URL (local): `http://127.0.0.1:8000`

### Verified working now

- `GET /health` - Health check, xem API đang chạy và các model đã load.
- `GET /models` - Liệt kê model available + model đã load.
- `POST /transcribe` - Nhận file audio và trả transcript bằng Whisper.

`POST /transcribe` hiện trả thêm `timestamps` theo một format duy nhất: mỗi item có `start` và `text` (giây bắt đầu nói nội dung nào).

### Available when model is loaded

- `POST /summarize` - Tóm tắt văn bản (cần model `summarization` load thành công).

### Quick test commands

```bash
# Health
curl -X GET "http://127.0.0.1:8000/health"

# Models
curl -X GET "http://127.0.0.1:8000/models"

# Transcribe
curl -X POST "http://127.0.0.1:8000/transcribe" \
	-F "file=@utils/data/audio/testaudio-vn.mp3"

# Summarize
curl -X POST "http://127.0.0.1:8000/summarize" \
	-H "Content-Type: application/json" \
	-d "{\"text\":\"Your long text here\"}"
```

## Configuration

Set these in `.env`:
- `HF_TOKEN` - Hugging Face token
- `HF_USERNAME` - Your username
- `WHISPER_MODEL_PATH` - Model path on HF Hub
- `WHISPER_DEVICE` - `auto` (default), `cuda`, `xpu`, `mps`, `directml`, or `cpu`
- `SUMMARIZATION_DEVICE` - `auto` (default), `cuda`, `xpu`, `mps`, `directml`, or `cpu`
- `API_PORT` - Server port (default: 8000)

## Run With GPU (Local)

Use this order when setting up GPU on Windows:

1. Check NVIDIA GPU + driver in terminal:

```powershell
nvidia-smi
```

2. Check whether your current PyTorch can see CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

3. If output shows a CPU build (for example `+cpu`) or `False`:

- For NVIDIA CUDA (recommended when available):

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

- For AMD/Intel/NVIDIA via DirectML backend on Windows:

```powershell
pip install torch-directml
```

4. Force app behavior via `.env`:

```env
WHISPER_DEVICE=auto
SUMMARIZATION_DEVICE=auto
```

Behavior:
- `*_DEVICE=auto`: use accelerator automatically in this order: `cuda` -> `xpu` -> `mps` -> `directml` -> `cpu`.
- `*_DEVICE=cuda`: force CUDA, fallback CPU if not found.
- `*_DEVICE=xpu`: force Intel XPU, fallback CPU if not found.
- `*_DEVICE=mps`: force Apple MPS, fallback CPU if not found.
- `*_DEVICE=directml`: force DirectML backend, fallback CPU if `torch-directml` is not installed.
- `*_DEVICE=cpu`: always run on CPU.

## Features

- Fine-tuned Whisper model for English & Vietnamese
- Auto GPU/CPU detection
- Modular design for multiple models
- Universal model pusher for HF Hub
- FastAPI REST server
- Support for MP3, WAV, FLAC formats

# Replicate API Guide - Streamland Whisper

## Model Information
- **Model**: shannonnonshan/streamland-whisper
- **URL**: https://replicate.com/shannonnonshan/streamland-whisper
- **API Token**: REPLICATE_API_TOKEN_REDACTED

## Run With Cog (Hugging Face model)

This repo's `predict.py` loads the model directly from the Hugging Face Hub: `shannonnonshan/streamland-whisper`.

If the Hugging Face repo is private or gated, export `HF_TOKEN` before building so Cog can download it during `cog push`.

1. Install `cog` if you don't have it:

```bash
pip install cog
```

2. Build the Cog runtime (first time):

```bash
cog build
```

3. Run a prediction with an audio file (example):

```bash
cog predict -i audio=/path/to/audio.wav -i language=en
```

From WSL, the push flow is:

```bash
export REPLICATE_API_TOKEN="your_replicate_token"
export HF_TOKEN="your_hf_token_if_needed"
cog push r8.im/shannonnonshan/streamland-whisper
```

Notes:
- `predict.py` already points to the HF model ID `shannonnonshan/streamland-whisper`, so no extra model changes are required.
- Cog uses a container runtime (Docker/Podman) to build and run environments. If you need a purely local, non-container workflow, run `python run.py` instead.
## Quick Start

### Python
```python
import replicate

output = replicate.run(
    "shannonnonshan/streamland-whisper:latest",
    input={
        "audio": "https://example.com/audio.wav",
        "language": "en",
        "task": "transcribe"
    }
)
print(output)
```

## Input Parameters
- `audio`: URL or path to audio file (required)
- `language`: Language code (optional, e.g., 'en', 'vi', 'fr')
- `task`: 'transcribe' or 'translate' (default: 'transcribe')

## Output
- `text`: Transcribed/translated text
- `language`: Detected language
- `segments`: Array of segments with timestamps

## References
- Replicate Docs: https://replicate.com/docs
- Model Page: https://replicate.com/shannonnonshan/streamland-whisper
