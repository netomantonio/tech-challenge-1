#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${FASE3_PYTHON:-$ROOT/.venv-fase3/bin/python}"
HOST="${FASE3_HOST:-127.0.0.1}"
PORT="${FASE3_PORT:-8010}"
export PYTHONPATH="$ROOT" FASE3_LLM_BACKEND=local HF_HUB_OFFLINE=1
export FASE3_ALLOWED_ORIGINS="${FASE3_ALLOWED_ORIGINS:-https://assistente-protocolos-fase3.pages.dev}"
if [[ -z "${HF_HOME:-}" && -f "$ROOT/.cache/fase3-hf-home.txt" ]]; then
  export HF_HOME="$(cat "$ROOT/.cache/fase3-hf-home.txt")"
fi
"$PYTHON" -m fase3.manage doctor --require-runtime
exec "$PYTHON" -m fase3 --backend local --host "$HOST" --port "$PORT" "$@"
