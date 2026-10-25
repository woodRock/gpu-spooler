#!/usr/bin/env bash
# Install gpu-spooler (a queue for jobs on SSH-reachable GPU servers): the `task` command.
#
#   curl -fsSL https://raw.githubusercontent.com/woodRock/gpu-spooler/main/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/woodRock/gpu-spooler/main/install.sh | bash -s -- --upgrade
#
# Downloads the prebuilt `task` for this machine from the latest GitHub release (a static binary on
# Linux), checks its SHA-256 and puts it in ~/.local/bin. No sudo, no Python, no Rust. A running
# daemon is stopped first (jobs keep running on the servers) and started again with the new
# version, which re-adopts them. An earlier install, Python (uv tool or a checkout's .venv) or
# Rust, is replaced; any other command called `task` is left alone unless --force.
#
# From a checkout (bash install.sh) it builds the checkout with cargo instead and links
# target/release/task, so a rebuild updates the command (and the daemon restarts itself).
set -euo pipefail

REPO="woodRock/gpu-spooler"
RELEASES="${GPU_SPOOLER_RELEASES:-https://github.com/$REPO/releases/latest/download}"
bin="${GPU_SPOOLER_BIN_DIR:-$HOME/.local/bin}"
state="$HOME/.gpu-spooler"

USAGE="Install gpu-spooler (the \`task\` command, for jobs on SSH-reachable GPU servers).

  curl -fsSL https://raw.githubusercontent.com/$REPO/main/install.sh | bash -s -- [OPTIONS]

Options:
  --upgrade     install the latest release over the current one
  --uninstall   remove it (~/.gpu-spooler, your config and task history, is kept)
  --force       replace a \`task\` command that isn't gpu-spooler's
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

# A checkout (bash install.sh from it, not piped from curl): build it.
here="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" 2>/dev/null && pwd)"
checkout=0
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "$here/Cargo.toml" ] && [ -d "$here/crates/gpu-spooler-cli" ]; then
  checkout=1
fi

case "$(uname -s)/$(uname -m)" in
  Linux/x86_64)                 target="x86_64-unknown-linux-musl" ;;
  Linux/aarch64|Linux/arm64)    target="aarch64-unknown-linux-musl" ;;
  Darwin/arm64)                 target="aarch64-apple-darwin" ;;
  Darwin/x86_64)                target="x86_64-apple-darwin" ;;
  *) die "no gpu-spooler build for $(uname -s) $(uname -m) (macOS and Linux, x86_64 or arm64; on Windows, use WSL)" ;;
esac

# ── what is installed now ─────────────────────────────────────────────────────

# Is this path gpu-spooler's `task` (any version, Python or Rust)? `task --version` says
# `task 0.x.y`; Taskwarrior's, for one, prints a bare version number.
ours() {
  [ -e "$1" ] || [ -L "$1" ] || return 1
  "$1" --version 2>/dev/null | grep -Eq '^task [0-9]+\.[0-9]+'
}

# A link left by the Python development install (dev/install-dev.sh): into a checkout's .venv, to
# an entry-point script that imports gpu_spooler (which still says so once the checkout has moved on).
python_dev_link() {
  [ -L "$1" ] || return 1
  local t; t="$(readlink "$1")"
  case "$t" in */.venv/bin/*) grep -q gpu_spooler "$t" 2>/dev/null ;; *) return 1 ;; esac
}

uv_installed() {
  command -v uv >/dev/null 2>&1 && uv tool list 2>/dev/null | awk '/^[^ -]/ { print $1 }' | grep -qx gpu-spooler
}

daemon_was_running=0
stop_daemon() {
  # Whichever version is installed: its PID file says. Jobs keep running on the servers.
  local pidf="$state/daemon.pid"
  [ -f "$pidf" ] || return 0
  local pid; pid="$(tr -dc 0-9 < "$pidf")"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 0
  daemon_was_running=1
  info "stopping the daemon (PID $pid); running jobs keep running"
  if [ "$dry" = 1 ]; then printf '  + kill %s\n' "$pid"; return 0; fi
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
  rm -f "$pidf"
}

remove_python_install() {
  if uv_installed; then
    info "removing the Python version (uv tool gpu-spooler)"
    run uv tool uninstall gpu-spooler >/dev/null
  fi
  # The old commands, gone with the Python version: now `task gpus` and `task tui`.
  for c in gpu-status tui; do
    if python_dev_link "$bin/$c"; then
      info "removing $bin/$c (now \`task ${c/gpu-status/gpus}\`)"
      run rm -f "$bin/$c"
    elif [ -e "$bin/$c" ] || [ -L "$bin/$c" ]; then
      warn "$bin/$c isn't gpu-spooler's Python version; left alone (gpu-spooler's is now \`task ${c/gpu-status/gpus}\`)"
    fi
  done
}

check_task_slot() {
  # A `task` that isn't gpu-spooler's (e.g. Taskwarrior) is left alone unless --force.
  local t="$bin/task"
  if [ -e "$t" ] || [ -L "$t" ]; then
    if ! ours "$t" && ! python_dev_link "$t"; then
      if [ "$force" = 0 ]; then
        info "$t is another program's \`task\`; left as it is:"
        info "    $(ls -l "$t" | sed 's/.* \(\/.*\)$/\1/')"
        info "re-run with --force to replace it with gpu-spooler's"
        exit 1
      fi
      warn "replacing $t (--force)"
    fi
  fi
}

# ── uninstall ─────────────────────────────────────────────────────────────────

if [ "$action" = "uninstall" ]; then
  bold "Removing gpu-spooler"
  stop_daemon
  remove_python_install
  if ours "$bin/task" || python_dev_link "$bin/task"; then
    run rm -f "$bin/task"
  elif [ -e "$bin/task" ]; then
    warn "$bin/task isn't gpu-spooler's; left alone"
  fi
  bold "Removed gpu-spooler"
  info "your config and task history are kept in $state"
  exit 0
fi

# ── install / upgrade ─────────────────────────────────────────────────────────

if [ "$checkout" = 1 ]; then
  bold "Development install from $here"
  command -v cargo >/dev/null 2>&1 || die "cargo is required to build a checkout (https://rustup.rs), or install a release with the curl one-liner"
  run cargo build --release --locked --manifest-path "$here/Cargo.toml" -p gpu-spooler-cli
  new="$here/target/release/task"
else
  bold "$([ "$action" = upgrade ] && echo Upgrading || echo Installing) gpu-spooler ($target)"
  command -v curl >/dev/null 2>&1 || die "curl is required"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  archive="gpu-spooler-$target.tar.gz"
  run curl -fsSL -o "$tmp/$archive" "$RELEASES/$archive" || die "could not download $RELEASES/$archive"
  run curl -fsSL -o "$tmp/$archive.sha256" "$RELEASES/$archive.sha256" || die "could not download $archive.sha256"
  if [ "$dry" = 0 ]; then
    want="$(awk '{ print $1 }' "$tmp/$archive.sha256")"
    if command -v sha256sum >/dev/null 2>&1; then got="$(sha256sum "$tmp/$archive" | awk '{ print $1 }')"
    else got="$(shasum -a 256 "$tmp/$archive" | awk '{ print $1 }')"; fi
    [ -n "$want" ] && [ "$want" = "$got" ] || die "$archive: SHA-256 mismatch (expected $want, got $got); nothing was installed"
    tar -xzf "$tmp/$archive" -C "$tmp" task
    "$tmp/task" -V >/dev/null 2>&1 || die "the downloaded task doesn't run on this machine"
    info "downloaded $("$tmp/task" -V), SHA-256 checked"
  fi
  new="$tmp/task"
fi

check_task_slot
stop_daemon
remove_python_install
run mkdir -p "$bin"
if [ "$checkout" = 1 ]; then
  run ln -sfn "$new" "$bin/task"
  info "linked $bin/task -> $new"
else
  # Atomically: a running `task` (or a daemon about to re-exec) never sees half a file.
  run cp "$new" "$bin/.task.new"
  run chmod 755 "$bin/.task.new"
  run mv -f "$bin/.task.new" "$bin/task"
  info "installed $bin/task"
fi
if [ "$daemon_was_running" = 1 ]; then
  run "$bin/task" daemon start >/dev/null
  info "daemon restarted with the new version; it re-adopts the running jobs"
fi

# ── PATH ──────────────────────────────────────────────────────────────────────
echo
if [[ ":$PATH:" != *":$bin:"* ]]; then
  rc="$HOME/.bashrc"; [ "${SHELL##*/}" = "zsh" ] && rc="$HOME/.zshrc"
  warn "$bin is not on your PATH. Add it, then open a new terminal:"
  echo "    echo 'export PATH=\"$bin:\$PATH\"' >> $rc"
  echo
fi
if [ "$action" = "upgrade" ]; then
  bold "Upgraded gpu-spooler"
else
  bold "Done. Try: task --help"
  [ "$checkout" = 1 ] || info "upgrade later: curl -fsSL https://raw.githubusercontent.com/$REPO/main/install.sh | bash -s -- --upgrade"
fi
