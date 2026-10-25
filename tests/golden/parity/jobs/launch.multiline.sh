D="$HOME/.gpu-spooler/jobs/7-654321"
mkdir -p "$D" || exit 1
cat > "$D/run.sh" <<'TS_EOF_7_654321'
D="$HOME/.gpu-spooler/jobs/7-654321"
source ~/.bashrc 2>/dev/null || true
ls /x > /dev/null 2>&1
cd /x || { echo "gpu-spooler: cannot cd to "/x >&2; echo 1 > "$D/rc.tmp"; mv "$D/rc.tmp" "$D/rc"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
:
( for i in 1 2; do
  echo $i
done
)
rc=$?
echo "$rc" > "$D/rc.tmp" && mv "$D/rc.tmp" "$D/rc"
exit $rc
TS_EOF_7_654321
ts_detach() {
  trap '' HUP     # like nohup: an ignored SIGHUP survives exec (macOS nohup
                  # refuses to run outside a console session)
  if command -v setsid >/dev/null 2>&1; then exec setsid "$@"
  else exec python3 -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' "$@"; fi
}
cd "$HOME"
# The commit the job will run, for run provenance (taura): HEAD and uncommitted changes.
# git may only be on a login shell's PATH (on ECS it's in /usr/pkg/bin).
command -v git >/dev/null 2>&1 || PATH="$(bash -lc 'printf %s "$PATH"' </dev/null 2>/dev/null):$PATH"
( cd /x 2>/dev/null && h=$(git rev-parse -q --verify HEAD 2>/dev/null)   && echo "TS_GIT $h $(git status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')"   && git status --porcelain --untracked-files=no 2>/dev/null | head -50 | sed 's/^.../TS_DIRTY /' )
ts_detach bash -l "$D/run.sh" > "$D/out.log" 2>&1 < /dev/null &
echo $! > "$D/pid"
echo "TS_PID $!"
