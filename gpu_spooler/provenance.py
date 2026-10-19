"""Optional run provenance: record each task as a taura run (github.com/woodRock/taura).

When `taura` is on PATH and the task's local repo has a taura ledger (a `ledger/` directory), the
daemon calls `taura record` when a task starts and finishes, and `task fetch --after ID` calls it
with the files it brought back. The record lands in the local repo's ledger/runs/, with the commit
the server actually ran (captured at launch), the host, GPUs and exit code. taura is never required:
without it, or with `[provenance] taura = "off"` in hosts.toml, nothing happens. A failing
`taura record` is logged and never affects the task.
"""

import os
import shutil
import subprocess
from datetime import datetime
from typing import Callable, Optional

RECORD_TIMEOUT = 60


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
    try:
        r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=RECORD_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"Task {row['id']}: taura record ({event}) failed: {e}")
        return False
    if r.returncode != 0:
        log(f"Task {row['id']}: taura record ({event}) exited {r.returncode}: "
            f"{(r.stderr or r.stdout).strip().splitlines()[-1:] or ''}")
        return False
    return True
