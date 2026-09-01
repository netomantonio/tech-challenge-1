#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${FASE3_SERVICE_ENV:-$ROOT/fase3/service.env}"
LOG_DIR="${FASE3_LOG_DIR:-$ROOT/.logs}"
LOG_FILE="$LOG_DIR/fase3-backend.log"
PID_FILE="$LOG_DIR/fase3-backend.pid"

if [[ -f "$CONFIG_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
  set +a
fi

REMOTE="${FASE3_GIT_REMOTE:-origin}"
BRANCH="${FASE3_GIT_BRANCH:-main}"
HOST="${FASE3_HOST:-127.0.0.1}"
PORT="${FASE3_PORT:-8010}"
ORIGINS="${FASE3_ALLOWED_ORIGINS:-https://assistente-protocolos-fase3.pages.dev}"
ALLOW_REMOTE_TRAINING="${FASE3_ALLOW_REMOTE_TRAINING:-0}"
INSTANCE_NAME="${FASE3_INSTANCE_NAME:-Backend local}"

resolve_python() {
  if [[ -n "${FASE3_PYTHON:-}" ]]; then
    printf '%s\n' "$FASE3_PYTHON"
    return
  fi
  if [[ -x "$ROOT/.venv-fase3/bin/python" ]]; then
    printf '%s\n' "$ROOT/.venv-fase3/bin/python"
    return
  fi
  local candidate
  for candidate in python python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      "$candidate" -c 'import sys; print(sys.executable)'
      return
    fi
  done
  echo "Erro: nenhum interpretador Python foi encontrado." >&2
  exit 1
}

PYTHON="$(resolve_python)"

is_fase3_process() {
  local pid="$1" command_line
  command_line="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  [[ "$command_line" == *"-m fase3"* || "$command_line" == *"fase3/start.sh"* ]]
}

stop_backend() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "Backend: nenhum PID registrado em $PID_FILE."
    return
  fi

  local pid
  pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
    echo "Backend: PID antigo ou inexistente; removendo registro."
    rm -f "$PID_FILE"
    return
  fi
  if ! is_fase3_process "$pid"; then
    echo "Erro: o PID $pid nao pertence ao backend da Fase 3; processo preservado." >&2
    exit 1
  fi

  echo "Backend: encerrando PID $pid..."
  kill -TERM "$pid"
  local attempt
  for attempt in {1..30}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "Backend: encerrado."
      return
    fi
    sleep 1
  done

  echo "Backend: encerramento gracioso expirou; finalizando PID $pid."
  kill -KILL "$pid"
  rm -f "$PID_FILE"
}

start_backend() {
  mkdir -p "$LOG_DIR"
  : > "$LOG_FILE"

  local -a args=(--no-open-browser --instance-name "$INSTANCE_NAME")
  if [[ "$ALLOW_REMOTE_TRAINING" == "1" ]]; then
    args+=(--allow-remote-training)
  fi

  echo "Backend: iniciando em $HOST:$PORT..."
  nohup env \
    FASE3_PYTHON="$PYTHON" \
    FASE3_HOST="$HOST" \
    FASE3_PORT="$PORT" \
    FASE3_ALLOWED_ORIGINS="$ORIGINS" \
    FASE3_ALLOW_REMOTE_TRAINING="$ALLOW_REMOTE_TRAINING" \
    FASE3_INSTANCE_NAME="$INSTANCE_NAME" \
    bash "$ROOT/fase3/start.sh" "${args[@]}" \
    > "$LOG_FILE" 2>&1 &

  local pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"

  local health_url="http://127.0.0.1:$PORT/api/capabilities"
  local attempt
  for attempt in {1..90}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Erro: o backend encerrou durante a inicializacao." >&2
      tail -n 80 "$LOG_FILE" >&2 || true
      exit 1
    fi
    if "$PYTHON" - "$health_url" >/dev/null 2>&1 <<'PY'
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
    then
      echo "Backend: pronto em $health_url (PID $pid)."
      echo "Log: $LOG_FILE"
      return
    fi
    sleep 1
  done

  echo "Erro: o backend nao ficou pronto em 90 segundos." >&2
  tail -n 80 "$LOG_FILE" >&2 || true
  exit 1
}

cd "$ROOT"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Erro: existem alteracoes versionadas locais. Commit, descarte ou guarde-as antes de atualizar." >&2
  git status --short >&2
  exit 1
fi

echo "Git: verificando $REMOTE/$BRANCH..."
git fetch "$REMOTE" "$BRANCH"
if [[ "$(git branch --show-current)" != "$BRANCH" ]]; then
  git switch "$BRANCH"
fi

PREVIOUS_REVISION="$(git rev-parse HEAD)"
stop_backend
if ! git pull --ff-only "$REMOTE" "$BRANCH"; then
  echo "Erro: a atualizacao falhou; reiniciando a versao disponivel no disco." >&2
  start_backend
  exit 1
fi

CURRENT_REVISION="$(git rev-parse HEAD)"
if [[ "$PREVIOUS_REVISION" != "$CURRENT_REVISION" ]] && \
  ! git diff --quiet "$PREVIOUS_REVISION" "$CURRENT_REVISION" -- \
    requirements.txt requirements-fase3.txt; then
  echo "Dependencias: arquivos de requisitos mudaram; atualizando ambiente..."
  "$PYTHON" -m pip install -r requirements.txt -r requirements-fase3.txt
fi
start_backend
