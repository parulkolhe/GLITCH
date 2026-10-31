#!/usr/bin/env bash
# GLITCH: one-command start (macOS / Linux). Creates .venv on first run, then serves the game.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [ ! -x .venv/bin/python ]; then
  echo "First run: creating .venv and installing the game server (no TensorFlow needed)…"
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r backend/requirements.txt
fi
PORT="${GLITCH_PORT:-8001}"
URL="http://127.0.0.1:$PORT"
echo "GLITCH is starting at $URL  (Ctrl+C to stop)"
( sleep 2; (command -v open >/dev/null && open "$URL") || (command -v xdg-open >/dev/null && xdg-open "$URL") || true ) &
exec .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port "$PORT"
