#!/usr/bin/env bash
set -euo pipefail

if [[ "${SEARCH_BUILD_ON_START:-true}" == "true" ]]; then
  echo "[INIT] Building search index..."
  python3 scripts/build_search_index.py || echo "[WARN] build_search_index.py failed; continuing startup"
fi

exec uvicorn api.server:app --host 0.0.0.0 --port "${PORT:-8080}"
