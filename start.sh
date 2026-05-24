#!/usr/bin/env bash
set -euo pipefail

# If users mounted a directory at /app/embeddings.faiss by mistake, switch to a safe runtime path.
if [[ -d "/app/embeddings.faiss" ]]; then
  export FAISS_INDEX_PATH="/app/runtime/embeddings.faiss"
  export FAISS_METADATA_PATH="/app/runtime/embeddings.meta.json"
  mkdir -p /app/runtime
  echo "[WARN] /app/embeddings.faiss is a directory. Using ${FAISS_INDEX_PATH} instead."
fi

if [[ "${SEARCH_BUILD_ON_START:-true}" == "true" ]]; then
  echo "[INIT] Building search index..."
  python3 scripts/build_search_index.py || echo "[WARN] build_search_index.py failed; continuing startup"
fi

exec uvicorn api.server:app --host 0.0.0.0 --port "${PORT:-8080}"
