"""Optional run provenance: record each task as a taura run (github.com/woodRock/taura).

When `taura` is on PATH and the task's local repo has a taura ledger (a `ledger/` directory), the
daemon calls `taura record` when a task starts and finishes, and `task fetch --after ID` calls it
with the files it brought back. The record lands in the local repo's ledger/runs/, with the commit
the server actually ran (captured at launch), the host, GPUs and exit code. taura is never required:
without it, or with `[provenance] taura = "off"` in hosts.toml, nothing happens. A failing
`taura record` is logged and never affects the task.
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime
from typing import Callable, Optional

RECORD_TIMEOUT = 60
ENV_MARK = "TAURA_ENV "

# The environment probe: a copy of taura's envprobe.PROBE (keep in step). It runs in the job's own
# interpreter on the server and prints one JSON line: Python, platform, GPUs, packages with versions
# and, for git or editable installs, their commits.
PROBE = r'''
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
'''


def run_id(row) -> str:
    """Stable per task attempt: <submit date>-spooler-<task id>[-<attempt>]."""
    day = datetime.fromtimestamp(row["submitted_at"]).strftime("%Y-%m-%d")
    attempt = (row["retry_count"] or 0) + 1
    return f"{day}-spooler-{row['id']}" + (f"-{attempt}" if attempt > 1 else "")


def ledger_repo(row) -> Optional[str]:
    """The local git repo the task came from, if it keeps a taura ledger."""
    start = row["repo_root"] or row["working_dir"]
    if not start or not os.path.isdir(start):
        return None
    r = subprocess.run(["git", "-C", start, "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    top = r.stdout.strip() if r.returncode == 0 else start
    return top if os.path.isdir(os.path.join(top, "ledger")) else None


def enabled(row, setting: str = "auto") -> bool:
    """Will this task be recorded with taura? (taura installed, repo keeps a ledger, not turned off)"""
    return setting != "off" and shutil.which("taura") is not None and ledger_repo(row) is not None


def probe_command(command: str) -> str:
    """How to run the probe like the job's command: its VAR=value prefix and its python (python3 if the
    command doesn't start with one, e.g. bash -c or torchrun)."""
    try:
        words = shlex.split(command)
    except ValueError:
        words = []
    env = []
    while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0]):
        env.append(words.pop(0))
    py = words[0] if words and re.fullmatch(r"python(\d+(\.\d+)*)?", os.path.basename(words[0])) else "python3"
    return " ".join(shlex.quote(w) for w in [*env, py])


def env_from_log(path: Optional[str]) -> Optional[dict]:
    """The environment a job reported (its last TAURA_ENV line), if any."""
    try:
        with open(path or "", errors="replace") as f:
            lines = [line[len(ENV_MARK):] for line in f if line.startswith(ENV_MARK)]
    except OSError:
        return None
    try:
        return json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        return None


def _field(row, key: str):
    return row[key] if key in row.keys() else None


def record_args(event: str, row, outputs: Optional[list[str]] = None) -> Optional[list[str]]:
    """The `taura record` command for this event, or None if this task isn't tracked."""
    repo = ledger_repo(row)
    if repo is None:
        return None
    args = ["-C", repo, "record", run_id(row), "--event", event, "--runner", "gpu-spooler",
            "--spooler-job", str(row["id"])]
    if event == "start":
        args += ["--command", row["command"]]
        wdir = os.path.realpath(row["working_dir"])
        if wdir != os.path.realpath(repo) and wdir.startswith(os.path.realpath(repo) + os.sep):
            args += ["--cwd", os.path.relpath(wdir, os.path.realpath(repo))]
        if row["server"]:
            args += ["--host", row["server"]]
        if row["gpu_indices"]:
            args += ["--gpus", row["gpu_indices"]]
        if row["started_at"]:
            args += ["--started", str(row["started_at"])]
        if _field(row, "git_commit"):
            args += ["--commit", row["git_commit"], "--dirty-count", str(_field(row, "git_dirty") or 0)]
    elif event == "finish":
        if row["exit_code"] is not None:
            args += ["--exit-code", str(row["exit_code"])]
        if row["finished_at"]:
            args += ["--finished", str(row["finished_at"])]
        if _field(row, "git_commit"):
            args += ["--commit", row["git_commit"], "--dirty-count", str(_field(row, "git_dirty") or 0)]
    elif event == "fetch":
        if not outputs:
            return None
        args += ["--outputs", *[os.path.relpath(os.path.realpath(p), os.path.realpath(repo)) for p in outputs]]
    return args


def notify(event: str, row, setting: str = "auto", outputs: Optional[list[str]] = None,
           log: Callable[[str], None] = lambda msg: None) -> bool:
    """Record a task event with taura, if enabled and available. True if a record was written."""
    if setting == "off":
        return False
    exe = shutil.which("taura")
    if exe is None:
        return False
    args = record_args(event, row, outputs)
    if args is None:
        return False
    env = env_from_log(_field(row, "log_file")) if event == "finish" else None
    env_file = None
    if env is not None:
        fd, env_file = tempfile.mkstemp(prefix="taura-env-", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(env, f)
        args += ["--env-json", env_file]
    try:
        r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=RECORD_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"Task {row['id']}: taura record ({event}) failed: {e}")
        return False
    finally:
        if env_file:
            os.unlink(env_file)
    if r.returncode != 0:
        log(f"Task {row['id']}: taura record ({event}) exited {r.returncode}: "
            f"{(r.stderr or r.stdout).strip().splitlines()[-1:] or ''}")
        return False
    return True
