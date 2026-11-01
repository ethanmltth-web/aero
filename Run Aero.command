#!/bin/bash
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  echo "First launch: installing Aero. This can take a few minutes."
  bash "./setup.sh" || exit 1
fi
export QT_API=pyside6
export PYTHONUNBUFFERED=1
exec ".venv/bin/python" -m aero_app
