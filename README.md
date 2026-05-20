# StreamLand AI - Multi-Model AI API

A comprehensive FastAPI-based service providing multiple AI models for speech recognition, text summarization, content moderation, and more. Designed for multilingual support (English & Vietnamese) with flexible deployment options.

## 🎯 Overview

StreamLand AI is a modular, production-ready API that orchestrates multiple state-of-the-art machine learning models:

| Model | Purpose | Status |
|-------|---------|--------|
| **Whisper** | Speech-to-Text (STT) | ✅ Active |
| **BART/ViT5** | Text Summarization | ✅ Active |
| **Moderation** | Content Moderation + Detoxification | ✅ Active |
| **Embeddings** | Semantic Search & Recommendations | ⏳ Planned |
| **Llama** | Chat & RAG | ⏳ Planned |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd streamland-ai

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your Hugging Face token and settings
```

### Running the API

```bash
# Start FastAPI server
python -m api.server
```

API available at: `http://127.0.0.1:8000`

### Quick Tests

```bash
# Health check
curl http://127.0.0.1:8000/health

# List models
curl http://127.0.0.1:8000/models

# Test Whisper locally
python run.py
```

## 📁 Project Structure

```
streamland-ai/
├── api/
│   ├── server.py                  # FastAPI app & model initialization
│   ├── model_registry.py          # Global model registry
│   ├── dependencies.py            # Dependency injection for models
│   └── endpoints/
│       ├── transcribe.py          # Speech-to-text endpoint
│       ├── summarize.py           # Text summarization endpoint
│       ├── moderation.py          # Content moderation endpoint
│       ├── chat.py                # LLM chat (future)
│       ├── search.py              # Semantic search (future)
│       └── recommend.py           # Recommendations (future)
│
├── models/
│   ├── base.py                    # Abstract model interface
│   ├── __init__.py                # Lazy-loading model imports
│   ├── whisper/
│   │   └── interface.py           # Whisper wrapper
│   ├── summarization/
│   │   └── interface.py           # Summarization wrapper
│   ├── moderation/
│   │   ├── interface.py           # Moderation engine
│   │   └── __init__.py
│   ├── embeddings/
│   │   └── interface.py           # Embeddings (future)
│   └── llama/
│       └── interface.py           # LLM (future)
│
├── utils/
│   ├── config.py                  # Centralized configuration
│   ├── model_loader.py            # Dynamic model loading
│   ├── model_pusher.py            # Push models to HF Hub
│   ├── replicate_client.py        # Replicate API integration
│   └── pipeline.py                # Model orchestration
│
├── requirements.txt               # Python dependencies
├── .env                           # Configuration (gitignored)
├── .env.example                   # Environment template
├── cog.yaml                       # Replicate Cog config
├── predict.py                     # Replicate prediction handler
├── run.py                         # Local test script
└── README.md                      # This file
```

## 🔌 API Endpoints

### Health & Status

```bash
GET /health
```
Returns server status, GPU info, and loaded models.

```bash
GET /models
```
Lists available and currently loaded models with metadata.

---

### 🎤 Speech-to-Text (Whisper)

```bash
POST /transcribe
```
**Input:** Audio file (MP3, WAV, FLAC)  
**Query params:** `language` (optional)

**Response:**
```json
{
  "status": "success",
  "data": {
    "filename": "audio.mp3",
    "result": {
      "text": "Full transcript here",
      "language": "en",
      "timestamps": [
        {"start": 0.0, "text": "Full transcript"},
        {"start": 1.2, "text": "here"}
      ]
    }
  }
}
```

**Auto-proxy to Replicate:** When `REPLICATE_USE=true`, automatically uses Replicate instead of loading model locally.

```bash
POST /transcribe/replicate
```
Explicitly use Replicate-hosted model.

---

### 📝 Text Summarization

```bash
POST /summarize
```

**Request:**
```json
{
  "text": "Long document text...",
  "max_length": 100,
  "min_length": 30
}
```

**Response:**
```json
{
  "status": "success",
  "text": "Original text...",
  "summary": "Summarized text..."
}
```

Auto-detects language (EN/VI) and uses appropriate model.

---

### 🛡️ Content Moderation

```bash
POST /moderation/text
```

**Request:**
```json
{
  "text": "Text to moderate",
  "rewrite": true
}
```

**Response:**
```json
{
  "status": "success",
  "text": "Text to moderate",
  "moderation": {
    "label": "REVIEW",
    "score": 0.62,
    "categories": ["insult"],
    "matched_spans": [...],
    "detoxified_text": "Message to moderate",
    "original_score": 0.62,
    "detox_score": 0.18,
    "rewrite_successful": true
  }
}
```

**Decision Labels:**
- `SAFE` (score < 0.55) - Content is acceptable
- `REVIEW` (0.55 ≤ score < 0.85) - Content needs review; detoxification offered
- `BLOCK` (score ≥ 0.85) - Content is harmful

---

## ⚙️ Configuration

Create `.env` file in project root:

```env
# Hugging Face Hub
HF_TOKEN=your_token_here
HF_USERNAME=your_username

# Whisper Configuration
WHISPER_MODEL_PATH=shannonnonshan/streamland-whisper-ct2
WHISPER_USE_HF=true
WHISPER_DEVICE=auto
WHISPER_COMPUTE_TYPE=float16

# Summarization Configuration
SUMMARIZATION_MODEL=shannonnonshan/bart-summarizer
SUMMARIZATION_USE_HF=true
SUMMARIZATION_DEVICE=auto

# Content Moderation Configuration
MODERATION_EN_MODEL=s-nlp/roberta_toxicity_classifier
MODERATION_VI_MODEL=cardiffnlp/twitter-xlm-roberta-base-offensive
MODERATION_FULL_MODEL=cardiffnlp/twitter-xlm-roberta-base-offensive
MODERATION_REWRITE_MODEL=s-nlp/bart-base-detox
MODERATION_EMBEDDING_MODEL=BAAI/bge-m3
MODERATION_BLOCK_THRESHOLD=0.85
MODERATION_REVIEW_THRESHOLD=0.55
MODERATION_GREYZONE_LOWER=0.40
MODERATION_GREYZONE_UPPER=0.70

# Replicate Integration (Optional)
REPLICATE_USE=false
REPLICATE_MODEL=shannonnonshan/streamland-whisper-ct2

# API Server
API_HOST=0.0.0.0
API_PORT=8000
```

---

## 🎯 Usage Examples

### Example 1: Transcribe Audio

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@path/to/audio.mp3" \
  -F "language=en"
```

### Example 2: Summarize Text

```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Long document content here...",
    "max_length": 150
  }'
```

### Example 3: Moderate Content

```bash
curl -X POST "http://127.0.0.1:8000/moderation/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Content to check",
    "rewrite": true
  }'
```

---

## 🛡️ Content Moderation Pipeline

The moderation system uses a 10-stage pipeline:

1. **Normalization** - Unicode normalization, lowercase, collapse repeated chars
2. **Lexicon Pre-check** - Fast keyword scan (early exit if clean)
3. **Language Split** - Detect English/Vietnamese/mixed using lingua
4. **Span Construction** - Generate spans for individual languages, boundaries, full text
5. **Toxicity Scoring** - Run parallel scorers (EN model for English, VI model for Vietnamese)
6. **Grey-zone RAG** - For ambiguous scores (0.40-0.70), retrieve similar examples and nudge
7. **Score Fusion** - Combine scores: 50% max + 30% mean + 20% full-sentence
8. **Decision Making** - SAFE/REVIEW/BLOCK based on thresholds
9. **Rewrite** (optional) - Detoxify REVIEW-level content using detox models
10. **Re-score** - Validate rewritten text

---

## 🖥️ GPU Setup (Windows)

### NVIDIA CUDA Setup

```powershell
# Check GPU availability
nvidia-smi

# Reinstall PyTorch with CUDA support
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Device Fallback Order

Set `*_DEVICE=auto` to use automatic device selection:
1. **CUDA** (NVIDIA GPU) - fastest
2. **XPU** (Intel GPU)
3. **MPS** (Apple Silicon)
4. **DirectML** (Windows GPU)
5. **CPU** (fallback)

---

## 🚢 Deployment

### Local Development

```bash
python -m api.server
```

### Docker

```bash
docker build -t streamland-ai:latest .
docker run -p 8000:8000 streamland-ai:latest
```

### Replicate

```bash
cog push r8.im/shannonnonshan/streamland-whisper-ct2
```

---

## 👨‍💻 Development

### Adding a New Model

1. Create interface in `models/<model_name>/interface.py`
2. Register in `utils/model_loader.py`
3. Add configuration in `utils/config.py`
4. Create endpoint in `api/endpoints/<model_name>.py`
5. Include router in `api/server.py`

### Testing

```bash
python run.py              # Test Whisper locally
curl http://127.0.0.1:8000/health  # Health check
```

---

## 🐛 Troubleshooting

### Import Errors

```bash
pip install -r requirements.txt
pip install lingua-language-detector sentence-transformers
```

### CUDA Not Found

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
echo "WHISPER_DEVICE=cpu" >> .env
```

### Model Download Failures

- Ensure `HF_TOKEN` is set for private models
- Check internet connection
- Verify token permissions

---

## 📚 References

- [Whisper GitHub](https://github.com/openai/whisper)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Hugging Face Hub](https://huggingface.co/)
- [Replicate Documentation](https://replicate.com/docs)

---

## 📝 License

Part of StreamLand AI initiative.

## 🤝 Contributing

1. Create feature branch
2. Make changes and test locally
3. Push and create pull request
4. Ensure all tests pass

---

## 📞 Support

For issues or questions, please open an issue on GitHub.
