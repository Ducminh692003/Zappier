#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f ".env" ]]; then
  set -a
  . ".env"
  set +a
fi

exec .venv/bin/python -m uvicorn local_app:app --host 127.0.0.1 --port "${PORT:-8000}"
