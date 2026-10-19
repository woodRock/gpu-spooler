#!/usr/bin/env bash
# Development install from a checkout (bash install.sh runs this): an editable .venv in the
# checkout, with task, gpu-status and tui linked into ~/.local/bin.
# Works on macOS (Homebrew Python) and Linux (conda, venv, or system Python).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the checkout (this file is in dev/)
VENV_DIR="$SCRIPT_DIR/.venv"
BIN_DIR="$HOME/.local/bin"

# ── Pick a Python interpreter ─────────────────────────────────────────────────
if   command -v python3.14 &>/dev/null; then PYTHON=python3.14
elif command -v python3.13 &>/dev/null; then PYTHON=python3.13
elif command -v python3.12 &>/dev/null; then PYTHON=python3.12
elif command -v python3.11 &>/dev/null; then PYTHON=python3.11
elif command -v python3.10 &>/dev/null; then PYTHON=python3.10
elif command -v python3    &>/dev/null; then PYTHON=python3
else echo "Error: Python 3.10+ required but not found." >&2; exit 1
fi

PY_VER=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using Python $PY_VER ($($PYTHON -c 'import sys; print(sys.executable)'))"

# ── Create / update virtual environment ──────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
fi

VENV_PIP="$VENV_DIR/bin/pip"
"$VENV_PIP" install --quiet --upgrade pip setuptools

echo "Installing gpu-spooler (with the tui extra) into venv ..."
if ! "$VENV_PIP" install --quiet -e "$SCRIPT_DIR[tui]"; then
    echo "  Could not install Textual; installing without tui."
    "$VENV_PIP" install --quiet -e "$SCRIPT_DIR"
fi

# ── Symlink entry points into ~/.local/bin ────────────────────────────────────
mkdir -p "$BIN_DIR"

for cmd in task gpu-status tui; do
    src="$VENV_DIR/bin/$cmd"
    dst="$BIN_DIR/$cmd"
    if [ -e "$src" ]; then
        ln -sf "$src" "$dst"
        echo "  Linked: $dst -> $src"
    else
        echo "  Warning: $src not found, skipping."
    fi
done

# ── PATH reminder ─────────────────────────────────────────────────────────────
echo ""
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo "  Add ~/.local/bin to your PATH. Put this in ~/.zshrc or ~/.bashrc:"
    echo ""
    echo '    export PATH="$HOME/.local/bin:$PATH"'
    echo ""
    echo "  Then restart your shell or run: source ~/.zshrc"
else
    echo "  Done! Run 'task -h', 'task gpus' or 'task tui' to get started."
fi
