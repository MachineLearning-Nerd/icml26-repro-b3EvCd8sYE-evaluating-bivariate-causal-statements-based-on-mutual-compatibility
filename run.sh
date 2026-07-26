#!/usr/bin/env bash
# Fixed reproduction command for every experiment node.  Inherited unchanged by
# every child; experimental variation lives in committed code (repro/config.py),
# never in this command or in environment variables.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "[run.sh] installing uv"
  curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --frozen
exec uv run --frozen python -m repro.run_all
