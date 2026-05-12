#!/usr/bin/env bash
# Run the infergate demo gateway.
# Usage: ./run_demo.sh [--port PORT]
set -euo pipefail

PORT=${2:-8080}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ── resolve OVH key ────────────────────────────────────────────────────────
if [[ -z "${INFERGATE_OVH_API_KEY:-}" ]]; then
    echo "[warn] INFERGATE_OVH_API_KEY not set — OVH backend will be skipped"
    echo "       Export it first:  export INFERGATE_OVH_API_KEY=<your-key>"
fi

# ── activate venv if present ───────────────────────────────────────────────
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
fi

# ── check dependencies ────────────────────────────────────────────────────
if ! python -c "import fastapi" 2>/dev/null; then
    echo "[error] fastapi not installed. Run: pip install 'infergate[demo]'"
    exit 1
fi
if ! python -c "import uvicorn" 2>/dev/null; then
    echo "[error] uvicorn not installed. Run: pip install 'infergate[demo]'"
    exit 1
fi

echo "[infergate] starting demo gateway on http://0.0.0.0:${PORT}"
exec uvicorn gateway:app \
    --app-dir "$SCRIPT_DIR" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info
