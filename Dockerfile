FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt ./requirements.txt
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision torchaudio \
    && python3 -m pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python3", "-m", "api.server"]