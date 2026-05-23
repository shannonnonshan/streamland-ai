FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    TRANSFORMERS_NO_TORCHAO=1 \
    PORT=8080

WORKDIR /app

ENV PYTHONPATH=/app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch CUDA 12.4
RUN python3 -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio

# Install Python dependencies
COPY requirements.txt .

RUN python3 -m pip install -r requirements.txt

# Copy source code
COPY . .

# Runtime entrypoint
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# HuggingFace cache directory
RUN mkdir -p /models/huggingface

EXPOSE 8080

# Start FastAPI
CMD ["/app/start.sh"]