D="$HOME/.gpu-spooler/jobs/4-ffffff"
mkdir -p "$D" || exit 1
cat > "$D/run.sh" <<'TS_EOF_4_ffffff'
D="$HOME/.gpu-spooler/jobs/4-ffffff"
source ~/.bashrc 2>/dev/null || true
ls "$HOME"/'work dir' > /dev/null 2>&1
cd "$HOME"/'work dir' || { echo "gpu-spooler: cannot cd to ""$HOME"/'work dir' >&2; echo 1 > "$D/rc.tmp"; mv "$D/rc.tmp" "$D/rc"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=''
:
( echo 'quoted' && ls "$HOME"
)
rc=$?
echo "$rc" > "$D/rc.tmp" && mv "$D/rc.tmp" "$D/rc"
exit $rc
TS_EOF_4_ffffff
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
( cd "$HOME"/'work dir' 2>/dev/null && h=$(git rev-parse -q --verify HEAD 2>/dev/null)   && echo "TS_GIT $h $(git status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')"   && git status --porcelain --untracked-files=no 2>/dev/null | head -50 | sed 's/^.../TS_DIRTY /' )
ts_detach bash -l "$D/run.sh" > "$D/out.log" 2>&1 < /dev/null &
echo $! > "$D/pid"
echo "TS_PID $!"
