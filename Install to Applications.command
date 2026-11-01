#!/bin/bash
# Copy Aero into the Applications folder and keep the bundled code up to date.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
APP="$ROOT/Aero.app"
RES="$APP/Contents/Resources"
DEST="/Applications/Aero.app"

echo "Packing Aero…"
rm -rf "$RES/aero_app"
mkdir -p "$RES"
cp -R "$ROOT/aero_app" "$RES/aero_app"
find "$RES/aero_app" -type d -name '__pycache__' -print0 | xargs -0 rm -rf
cp "$ROOT/requirements.txt" "$ROOT/setup.sh" "$RES/"
chmod +x "$APP/Contents/MacOS/Aero" "$RES/setup.sh"

echo "Installing to /Applications…"
rm -rf "$DEST"
ditto "$APP" "$DEST"
xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true

SUPPORT="$HOME/Library/Application Support/Aero"
if [ ! -x "$SUPPORT/.venv/bin/python" ]; then
  if [ -x "$ROOT/.venv/bin/python" ] && [ -d "$ROOT/.python" ]; then
    echo "Copying the Python tools into Application Support…"
    mkdir -p "$SUPPORT"
    ditto "$ROOT/.python" "$SUPPORT/.python"
    ditto "$ROOT/.tools" "$SUPPORT/.tools"
    ditto "$ROOT/.venv" "$SUPPORT/.venv"
    PYBIN="$(find "$SUPPORT/.python" -type f -path '*/bin/python3.12' | head -1)"
    ln -sfn "$PYBIN" "$SUPPORT/.venv/bin/python"
    ln -sfn python "$SUPPORT/.venv/bin/python3"
    ln -sfn python "$SUPPORT/.venv/bin/python3.12"
    printf 'home = %s\nimplementation = CPython\nuv = 0.13.6\nversion_info = 3.12\ninclude-system-site-packages = false\n' "$(dirname "$PYBIN")" > "$SUPPORT/.venv/pyvenv.cfg"
  else
    echo "First launch from Applications will download the Python tools."
  fi
fi

echo
echo "Aero is in Applications. Open it from Launchpad or the Applications folder."
if [ -t 0 ]; then
  echo
  read -r -p "Press Return to close…"
fi
