# StreamLand AI API Endpoints

Comprehensive API endpoints for transcription, summarization, and combined pipeline operations.

## Available Endpoints

### 1. Transcription (Whisper)
**Endpoint:** `POST /transcribe`

Transcribe audio files to text using OpenAI Whisper.

**Request:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.mp3"
```

**Response:**
```json
{
  "status": "success",
  "filename": "audio.mp3",
  "result": "Your transcribed text here..."
}
```

---

### 2. Summarization (FLAN-T5)
**Endpoint:** `POST /summarize`

Summarize text using Google FLAN-T5 model.

**Request Body:**
```json
{
  "text": "Your long text to summarize...",
  "max_length": 150,
  "min_length": 30,
  "use_rag": false,
  "rag_context_type": "general"
}
```

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long text to summarize...",
    "max_length": 150,
    "min_length": 30,
    "use_rag": true,
    "rag_context_type": "streamland_ai"
  }'
```

**Response:**
```json
{
  "status": "success",
  "original_length": 500,
  "summary": "Summarized text here...",
  "summary_length": 50,
  "compression_ratio": 0.1
}
```

**Parameters:**
- `text` (string, required): Text to summarize
- `max_length` (int, default: 150): Maximum length of summary
- `min_length` (int, default: 30): Minimum length of summary
- `use_rag` (bool, default: false): Use RAG context for better summarization
- `rag_context_type` (string): Type of RAG context
  - `"general"`: General summarization context
  - `"streamland_ai"`: StreamLand AI platform context
  - `"technical"`: Technical/AI/ML context

---

### 3. Combined Pipeline (Transcribe + Summarize)
**Endpoint:** `POST /pipeline/transcribe-summarize`

Transcribe audio and automatically summarize the transcription in one request.

**Request:**
```bash
curl -X POST "http://localhost:8000/pipeline/transcribe-summarize" \
  -F "file=@audio.mp3" \
  -F "max_summary_length=150" \
  -F "min_summary_length=30" \
  -F "use_rag=true" \
  -F "rag_context_type=streamland_ai"
```

**Form Parameters:**
- `file` (file, required): Audio file to transcribe
- `max_summary_length` (int, default: 150): Max summary length
- `min_summary_length` (int, default: 30): Min summary length
- `use_rag` (bool, default: false): Use RAG context
- `rag_context_type` (string): Type of RAG context

**Response:**
```json
{
  "status": "success",
  "filename": "audio.mp3",
  "transcription": "Full transcribed text...",
  "transcription_length": 500,
  "summary": "Summarized text...",
  "summary_length": 50,
  "compression_ratio": 0.1
}
```

---

### 4. Model Information

**Transcription Info:** `GET /transcribe/info`
```bash
curl "http://localhost:8000/transcribe/info"
```

**Summarization Info:** `GET /summarize/info`
```bash
curl "http://localhost:8000/summarize/info"
```

**Pipeline Info:** `GET /pipeline/info`
```bash
curl "http://localhost:8000/pipeline/info"
```

---

### 5. System Endpoints

**Health Check:** `GET /health`
```bash
curl "http://localhost:8000/health"
```

**List Available Models:** `GET /models`
```bash
curl "http://localhost:8000/models"
```

---

## RAG Context Types

The API supports three types of RAG (Retrieval-Augmented Generation) contexts:

1. **general**: Default general-purpose summarization context
2. **streamland_ai**: Context specific to StreamLand AI platform
3. **technical**: Technical/AI/ML specific context

Enable RAG context by setting `use_rag=true` in your request.

---

## Integration Example (Python)

```python
import requests
import json

# API Base URL
BASE_URL = "http://localhost:8000"

# 1. Summarize text
def summarize_text(text, use_rag=False):
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={
            "text": text,
            "max_length": 150,
            "min_length": 30,
            "use_rag": use_rag,
            "rag_context_type": "streamland_ai"
        }
    )
    return response.json()

# 2. Transcribe and summarize audio
def transcribe_and_summarize(audio_file_path, use_rag=False):
    with open(audio_file_path, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/pipeline/transcribe-summarize",
            files={"file": f},
            data={
                "max_summary_length": 150,
                "min_summary_length": 30,
                "use_rag": use_rag,
                "rag_context_type": "streamland_ai"
            }
        )
    return response.json()

# Usage
if __name__ == "__main__":
    # Test summarization
    text = "Long text to summarize..."
    result = summarize_text(text, use_rag=True)
    print(json.dumps(result, indent=2))
    
    # Test pipeline
    result = transcribe_and_summarize("audio.mp3", use_rag=True)
    print(json.dumps(result, indent=2))
```

---

## Configuration

Set these environment variables in `.env`:

```env
# Whisper
WHISPER_MODEL_PATH=shannonnonshan/streamland-whisper
WHISPER_USE_HF=true

# FLAN-T5 (Summarization)
FLAN_MODEL_PATH=shannonnonshan/flan-t5-small
FLAN_USE_HF=true

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
```

---

## Running the API Server

```bash
# Navigate to project directory
cd d:\streamland-ai

# Start the API server
python api/server.py
```

The API will be available at `http://localhost:8000`

Interactive API documentation: `http://localhost:8000/docs`

---

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (missing required fields, invalid model, etc.)
- `500`: Server error (model loading failed, inference error, etc.)

Error responses include a detail message:
```json
{
  "detail": "Error message describing what went wrong"
}
```
