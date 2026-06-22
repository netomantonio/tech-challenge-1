#!/usr/bin/env bash
set -euo pipefail

if [[ "${WORKERS_CI:-}" != "1" ]]; then
  echo "Este comando só pode publicar pelo Cloudflare Builds conectado ao GitHub." >&2
  exit 1
fi

bucket="cancer-mama-artifacts-preview"
commit_sha="${WORKERS_CI_COMMIT_SHA:?WORKERS_CI_COMMIT_SHA não informado}"

if ! npx wrangler r2 bucket info "$bucket" >/dev/null 2>&1; then
  npx wrangler r2 bucket create "$bucket"
fi

while IFS= read -r -d '' file; do
  npx wrangler r2 object put "$bucket/builds/$commit_sha/$file" --file "$file"
done < <(
  git ls-files -z \
    src/modelo_serving.json \
    resultados/fase2 \
    docs \
    'relatorio_tecnico_*.md' \
    'relatorio_tecnico_*.pdf'
)

(
  cd cloudflare/api
  uv run pywrangler deploy
)
