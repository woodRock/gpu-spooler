command -v git >/dev/null 2>&1 || PATH="$(bash -lc 'printf %s "$PATH"' </dev/null 2>/dev/null):$PATH"
command -v git >/dev/null 2>&1 || { echo "git is not installed" >&2; exit 3; }
P="$HOME"/'repos/my proj'
if [ ! -e "$P" ]; then
  mkdir -p "$P" && git init -q "$P" || exit 1
elif [ "$(git -C "$P" rev-parse --show-toplevel 2>/dev/null)" != "$(cd "$P" && pwd -P)" ]; then
  if [ -n "$(ls -A "$P")" ]; then echo "$P exists and is not a git repo" >&2; exit 4; fi
  git init -q "$P" || exit 1
fi
echo READY
