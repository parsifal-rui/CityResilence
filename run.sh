#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "未找到 .venv。请先创建并安装依赖："
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
if [[ $# -eq 0 ]]; then
  exec "$PY" "$ROOT/src/deepseek_event_schema.py"
fi
if [[ "$1" == src/* ]] || [[ "$1" == /* ]]; then
  exec "$PY" "$@"
else
  exec "$PY" "$ROOT/src/${1%.py}.py" "${@:2}"
fi
