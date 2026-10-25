command -v git >/dev/null 2>&1 || PATH="$(bash -lc 'printf %s "$PATH"' </dev/null 2>/dev/null):$PATH"
command -v git >/dev/null 2>&1 || { echo "STATE nogit"; exit 0; }
P="$HOME"/'repos/my proj'
if [ ! -e "$P" ]; then echo "STATE missing"
elif [ "$(git -C "$P" rev-parse --show-toplevel 2>/dev/null)" != "$(cd "$P" && pwd -P)" ]; then
  echo "STATE notgit"
else
  echo "STATE ok"
  echo "TREE $(git -C "$P" rev-parse -q --verify 'HEAD^{tree}' 2>/dev/null)"
  echo "HEAD $(git -C "$P" rev-parse -q --verify HEAD 2>/dev/null)"
  echo "DIRTY $(git -C "$P" status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')"
fi
