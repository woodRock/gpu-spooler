
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
