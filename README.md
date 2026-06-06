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

## 🔌 API Endpoints

### Health & Status

GET /health — returns server status, GPU info, and loaded models.

GET /models — lists available and currently loaded models with metadata.

---

### 🎤 Speech-to-Text (Whisper)

POST /transcribe

Input (multipart/form-data): either an uploaded `file` or a `file_url` form field. Optional `language` form field.

The endpoint returns a streaming NDJSON `application/x-ndjson` response with progress messages. The final message contains `status: "success"` and `data.result` with the transcription output returned by the Whisper model.

Example:

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@path/to/audio.mp3" \
  -F "language=en"
```

---

### 📝 Text Summarization

POST /summarize

Request JSON body:

```json
{
  "text": "Long document text..."
}
```

Response JSON:

```json
{
  "status": "success",
  "data": {
    "input_length": 1234,
    "summary": "Summarized text..."
  },
  "error": null
}
```

Note: the summarization endpoint accepts plain text only (`text`) and will return an error for unsupported languages.

---

### 🔎 Search (Embeddings)

POST /search

Query parameters:
- `query` (string, required)
- `top_k` (int, optional, default 5)

Response JSON contains `results` — a list of matched items (metadata and score).

Example:

```bash
curl -X POST "http://127.0.0.1:8000/search?query=education&top_k=5"
```

---

### 💬 Chat

POST /chat

Request JSON:

```json
{
  "message": "What is past perfect?",
  "history": [{"role":"user","msg":"Hello"}],
  "exclude_ids": [],
  "top_k": 5
}
```

Response JSON includes the model `response` and any `retrieved_ids` pulled from the search index.

Example:

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"What is past perfect?","history":[],"exclude_ids":[],"top_k":5}'
```

---

### 🛡️ Content Moderation

POST /moderation/text

Request JSON:

```json
{
  "text": "Text to check"
}
```

Response JSON contains a `moderation` object with `status`, `score`, `toxic_word`, and `categories`.

POST /moderation/image

Currently returns HTTP 501 (not implemented).

Decision labels used by the moderation pipeline:
- `SAFE` — content acceptable (score < `MODERATION_REVIEW_THRESHOLD`)
- `REVIEW` — ambiguous content (between review and block thresholds)
- `BLOCK` — content flagged as harmful (score ≥ `MODERATION_BLOCK_THRESHOLD`)

---
      "language": "en",
      "timestamps": [
        {"start": 0.0, "text": "Full transcript"},
        {"start": 1.2, "text": "here"}
      ]
    }
  }
}
```

Transcription runs locally on the machine's GPU when CUDA is available, so on Google Cloud you can point this service at a GPU VM or GPU-enabled container runtime and keep inference entirely inside your own cloud account.

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
  "text": "Text to moderate"
}
```

**Response:**
```json
{
  "status": "success",
  "text": "Text to moderate",
  "moderation": {
    "status": "REVIEW",
    "toxic_word": ["stupid"],
    "label": "REVIEW",
    "score": 0.62,
    "categories": ["insult"]
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
<<<<<<< HEAD
  -H "Content-Type: application/json" \
  -d '{
    "text": "Long document content here...",
    "max_length": 150
  }'
=======
	-H "Content-Type: application/json" \
	-d "{\"text\":\"Your long text here\"}"

# Chatbot
curl -X POST "http://127.0.0.1:8000/chat" \
    -H "Content-Type: application/json" \
    -d "{\"message\":\"What is past perfect?\",\"history\":[{\"role\":\"user\",\"msg\":\"Hello\"}],\"exclude_ids\":[],\"top_k\":5}"
>>>>>>> origin/vythanh-4-ai
```

### Example 3: Moderate Content

```bash
curl -X POST "http://127.0.0.1:8000/moderation/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Content to check"
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

### Google Cloud GPU

Run the API on a GPU-backed Google Cloud VM or GKE node pool. The recommended path is a container image built from this repo, then run it with NVIDIA GPU support enabled.

1. Create an Artifact Registry repository.

```bash
gcloud services enable compute.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud artifacts repositories create streamland-ai \
  --repository-format=docker \
  --location=asia-southeast1 \
  --description="StreamLand AI images"
```

2. Build and push the image.

```bash
gcloud builds submit --config cloudbuild.yaml .
```

3. Deploy on a GPU-backed Compute Engine VM.

```bash
gcloud compute instances create streamland-ai-gpu \
  --zone=asia-southeast1-b \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --restart-on-failure \
  --boot-disk-size=100GB \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud
```

After the VM is ready, install Docker, NVIDIA drivers, and the Google Cloud GPU container runtime, then run:

```bash
docker run --gpus all -p 8000:8000 \
  -e WHISPER_DEVICE=cuda \
  -e WHISPER_COMPUTE_TYPE=float16 \
  -e SUMMARIZATION_DEVICE=cuda \
  -e HF_TOKEN=$HF_TOKEN \
  asia-southeast1-docker.pkg.dev/$PROJECT_ID/streamland-ai/streamland-ai:latest
```

If you prefer GKE, use the same image and request a GPU node pool with NVIDIA drivers enabled.

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

### Google Cloud GPU

- Use a GPU image or install NVIDIA drivers/CUDA on the VM
- Set `WHISPER_DEVICE=auto` or `WHISPER_DEVICE=cuda`
- Set `SUMMARIZATION_DEVICE=auto` or `SUMMARIZATION_DEVICE=cuda`
- Verify Docker has GPU access with `docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 nvidia-smi`

---

## 📚 References

- [Whisper GitHub](https://github.com/openai/whisper)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Hugging Face Hub](https://huggingface.co/)
- [Google Cloud Compute Engine GPU](https://cloud.google.com/compute/docs/gpus)

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
