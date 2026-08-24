#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${FASE3_VENV:-$ROOT/.venv-fase3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ALIAS="${FASE3_SETUP_MODEL:-qwen2.5-1.5b}"
HF_HOME="${FASE3_HF_HOME:-${HF_HOME:-$ROOT/.cache/huggingface}}"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt" -r "$ROOT/requirements-fase3.txt"
mkdir -p "$ROOT/.cache"
printf '%s' "$HF_HOME" > "$ROOT/.cache/fase3-hf-home.txt"
export PYTHONPATH="$ROOT" HF_HOME
unset HF_HUB_OFFLINE || true
"$VENV/bin/python" -m fase3.manage install --alias "$MODEL_ALIAS"
echo "Setup concluido. Execute: bash fase3/start.sh"
