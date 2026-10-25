D="$HOME/.gpu-spooler/jobs/6-123456"
mkdir -p "$D" || exit 1
cat > "$D/taura_probe.py" <<'TS_PROBE_6_123456'
import json, os, platform, subprocess, sys
try:
    from importlib import metadata
except ImportError:
    metadata = None

def git(path, *args):
    try:
        r = subprocess.run(["git", "-C", path] + list(args), capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""

packages = {}
for d in (metadata.distributions() if metadata else []):
    try:
        name = (d.metadata["Name"] or "").lower()
    except Exception:
        continue
    if not name or name in packages:
        continue
    version, commit, dirty = d.version, "", False
    try:
        url = json.loads(d.read_text("direct_url.json") or "null") or {}
    except Exception:
        url = {}
    commit = (url.get("vcs_info") or {}).get("commit_id") or ""
    if (url.get("dir_info") or {}).get("editable") and url.get("url", "").startswith("file://"):
        path = url["url"][len("file://"):]
        commit = git(path, "rev-parse", "HEAD") or commit
        dirty = bool(git(path, "status", "--porcelain", "--untracked-files=no"))
        version += " editable"
    if commit:
        version += " @" + commit[:12] + ("+dirty" if dirty else "")
    packages[name] = version

gpus = []
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=20).stdout
    gpus = [line.strip() for line in out.splitlines() if line.strip()]
except Exception:
    pass

print(json.dumps({"python": platform.python_version(), "executable": sys.executable,
                  "platform": platform.platform(), "machine": platform.machine(),
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpus": gpus,
                  "packages": dict(sorted(packages.items()))}, separators=(",", ":")))
TS_PROBE_6_123456
cat > "$D/run.sh" <<'TS_EOF_6_123456'
D="$HOME/.gpu-spooler/jobs/6-123456"
source ~/.bashrc 2>/dev/null || true
ls '/vol/me/it'"'"'s' > /dev/null 2>&1
cd '/vol/me/it'"'"'s' || { echo "gpu-spooler: cannot cd to "'/vol/me/it'"'"'s' >&2; echo 1 > "$D/rc.tmp"; mv "$D/rc.tmp" "$D/rc"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
:
PYTHONPATH=src python3 "$D/taura_probe.py" 2>/dev/null | sed "s/^/TAURA_ENV /" || true
( PYTHONPATH=src python3 -m x
)
rc=$?
echo "$rc" > "$D/rc.tmp" && mv "$D/rc.tmp" "$D/rc"
exit $rc
TS_EOF_6_123456
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
( cd '/vol/me/it'"'"'s' 2>/dev/null && h=$(git rev-parse -q --verify HEAD 2>/dev/null)   && echo "TS_GIT $h $(git status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')"   && git status --porcelain --untracked-files=no 2>/dev/null | head -50 | sed 's/^.../TS_DIRTY /' )
ts_detach bash -l "$D/run.sh" > "$D/out.log" 2>&1 < /dev/null &
echo $! > "$D/pid"
echo "TS_PID $!"
