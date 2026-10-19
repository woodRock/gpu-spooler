#!/usr/bin/env bash
# Install gpu-spooler (a queue for jobs on SSH-reachable GPU servers).
#
#   curl -fsSL https://raw.githubusercontent.com/woodRock/gpu-spooler/main/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/woodRock/gpu-spooler/main/install.sh | bash -s -- --upgrade
#
# gpu-spooler is installed from GitHub with uv (https://docs.astral.sh/uv/) into its own isolated
# environment, with its commands (task gpu-status tui) linked into ~/.local/bin. uv is installed first if
# missing, and uv downloads Python 3.11 if the machine doesn't have it. Nothing needs sudo.
# The same installer, with the same options, is in kete, ruru, taura and gpu-spooler.

# Run from a checkout (bash install.sh): the development install, an editable .venv in the
# checkout (dev/install-dev.sh). Piped from curl, there is no checkout: install as a uv tool.
here="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" 2>/dev/null && pwd)"
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "$here/dev/install-dev.sh" ] && [ -f "$here/pyproject.toml" ]; then
  exec bash "$here/dev/install-dev.sh" "$@"
fi

set -euo pipefail

TOOL="gpu-spooler"
SPEC="gpu-spooler[tui] @ git+https://github.com/woodRock/gpu-spooler"
COMMANDS="task gpu-status tui"
PYTHON="3.11"

USAGE="Install gpu-spooler (a queue for jobs on SSH-reachable GPU servers).

  curl -fsSL https://raw.githubusercontent.com/woodRock/gpu-spooler/main/install.sh | bash -s -- [OPTIONS]

Options:
  --upgrade     reinstall at the latest main
  --uninstall   remove it
  --force       replace commands of the same name installed another way (pipx, pip, a checkout)
  --dry-run     print what would run, run nothing"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

action="install"; dry=0; force=0
for arg in "$@"; do
  case "$arg" in
    --upgrade)   action="upgrade" ;;
    --uninstall) action="uninstall" ;;
    --force)     force=1 ;;
    --dry-run)   dry=1 ;;
    -h|--help)   printf '%s\n' "$USAGE"; exit 0 ;;
    *)           die "unknown option $arg (try --help)" ;;
  esac
done

run() {
  if [ "$dry" = 1 ]; then printf '  + %s\n' "$*"; else "$@"; fi
}

case "$(uname -s)" in
  Darwin|Linux) ;;
  *) die "macOS and Linux only (on Windows, use WSL)" ;;
esac

# ── uv ────────────────────────────────────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
  for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    if [ -x "$d/uv" ]; then export PATH="$d:$PATH"; break; fi
  done
fi
if ! command -v uv >/dev/null 2>&1; then
  [ "$action" = "uninstall" ] && die "uv is not installed, so neither is $TOOL"
  bold "Installing uv"
  command -v curl >/dev/null 2>&1 || die "curl is required"
  if [ "$dry" = 1 ]; then
    info "+ curl -LsSf https://astral.sh/uv/install.sh | sh"
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

bin=$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")

# ── gpu-spooler ─────────────────────────────────────────────────────────────────────────────────
flags=()
case "$action" in
  uninstall)
    bold "Removing $TOOL"
    run uv tool uninstall "$TOOL"
    bold "Removed $TOOL"
    exit 0 ;;
  upgrade)
    bold "Upgrading $TOOL"
    flags=(--reinstall) ;;
  install)
    bold "Installing $TOOL" ;;
esac

# Commands of the same name not installed by uv (pipx, pip --user, a checkout's own script):
# left alone unless --force, since they may be someone's development install.
if ! uv tool list 2>/dev/null | awk '/^[^ -]/ { print $1 }' | grep -qx "$TOOL"; then
  existing=""
  for c in $COMMANDS; do
    if [ -e "$bin/$c" ] || [ -L "$bin/$c" ]; then
      existing+="      $bin/$c"
      [ -L "$bin/$c" ] && existing+=" -> $(readlink "$bin/$c")"
      existing+=$'\n'
    fi
  done
  if [ -n "$existing" ]; then
    if [ "$force" = 0 ]; then
      info "already installed another way, left as it is:"
      printf '%s' "$existing"
      info "re-run with --force to replace it with this installer's copy"
      exit 0
    fi
    flags+=(--force)
  fi
fi

run uv tool install --quiet --python "$PYTHON" "${flags[@]+"${flags[@]}"}" "$SPEC" \
  || die "$TOOL did not install"
info "commands: $COMMANDS"

# ── PATH ──────────────────────────────────────────────────────────────────────────────────
echo
if [[ ":$PATH:" != *":$bin:"* ]]; then
  rc="$HOME/.bashrc"; [ "${SHELL##*/}" = "zsh" ] && rc="$HOME/.zshrc"
  warn "$bin is not on your PATH. Add it, then open a new terminal:"
  echo "    echo 'export PATH=\"$bin:\$PATH\"' >> $rc"
  echo
fi
first=${COMMANDS%% *}
if [ "$action" = "upgrade" ]; then
  bold "Upgraded $TOOL"
else
  bold "Done. Try: $first --help"
  info "upgrade later: curl -fsSL https://raw.githubusercontent.com/woodRock/$TOOL/main/install.sh | bash -s -- --upgrade"
fi
