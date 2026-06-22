#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f data/cancer_mama.csv ]]; then
  uv run --project cloudflare/api --group test python data/download_datasets.py
fi

PYTHONPATH=. uv run --project cloudflare/api --group test python tests/test_fase2.py
PYTHONPATH=. uv run --project cloudflare/api --group test python tests/test_cloudflare.py
uv run --project cloudflare/api --group test python scripts/export_serving_model.py --check
(
  cd cloudflare/api
  uv run pywrangler deploy --dry-run --outdir ../../dist/worker
)
