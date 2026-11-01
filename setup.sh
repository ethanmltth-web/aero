#!/bin/bash
# Install a private Python and the Aero libraries.
# Usage: setup.sh [folder]
# With no folder, it installs next to this script.
set -euo pipefail
if [ "${1:-}" != "" ]; then
  ROOT="$1"
else
  ROOT="$(cd "$(dirname "$0")" && pwd)"
fi
mkdir -p "$ROOT"
cd "$ROOT"
mkdir -p "$ROOT/.tools" "$ROOT/.python"

if [ ! -x "$ROOT/.tools/uv" ]; then
  echo "Downloading the Python installer…"
  curl -LsSf "https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz" -o /tmp/aero-uv.tar.gz
  tar -xzf /tmp/aero-uv.tar.gz -C "$ROOT/.tools" --strip-components=1
fi

REQ="$(cd "$(dirname "$0")" && pwd)/requirements.txt"
if [ ! -f "$REQ" ]; then
  REQ="$ROOT/requirements.txt"
fi

export UV_PYTHON_INSTALL_DIR="$ROOT/.python"
echo "Installing Python 3.12…"
"$ROOT/.tools/uv" python install 3.12
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "Creating the app environment…"
  "$ROOT/.tools/uv" venv --python 3.12 "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install -U pip
"$ROOT/.venv/bin/python" -m pip install -r "$REQ"
echo
echo "Aero is ready."
