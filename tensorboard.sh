#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.tools/murre-env/bin/python}"
LOGDIR="${LOGDIR:-$ROOT/output/murre-normal-prior-b16/tensorboard}"
PORT="${PORT:-6006}"
exec "$PYTHON_BIN" -m tensorboard.main --logdir "$LOGDIR" --port "$PORT" "$@"
