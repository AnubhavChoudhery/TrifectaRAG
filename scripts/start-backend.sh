#!/usr/bin/env bash
# Start the TrifectaRAG FastAPI backend on port 8000.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Prefer Anaconda Python (system python3 may be too old or missing uvicorn)
if [ -x "/opt/anaconda3/bin/python" ]; then
  PYTHON="/opt/anaconda3/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "No Python found. Install Python 3.10+ and run: pip install -r requirements.txt"
  exit 1
fi

echo "Using Python: $PYTHON"
echo "Starting backend on http://127.0.0.1:8000"
echo "Note: first start loads ML models in the background (~1 min). /health works immediately."
echo ""

exec "$PYTHON" -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
