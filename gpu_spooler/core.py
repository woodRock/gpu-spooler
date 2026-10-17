"""
Shared task-spooler operations: the queue database, daemon process control,
and every action the front ends offer (submit, kill, resubmit, clear, ...).

Nothing here prints. The `task` CLI and the `tui` app are thin layers that
call these functions and render the results.
"""

import concurrent.futures
import json
import os
import shlex
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from gpu_spooler import jobs, repo
from gpu_spooler.config import STATE_DIR, Config, ConfigError, Host, load_config
from gpu_spooler.transport import Transport

# ── Paths ─────────────────────────────────────────────────────────────────────

DB_PATH          = STATE_DIR / "tasks.db"
PID_FILE         = STATE_DIR / "daemon.pid"
DAEMON_LOG       = STATE_DIR / "daemon.log"
LOGS_DIR         = STATE_DIR / "logs"
SNAPSHOT_PATH    = STATE_DIR / "gpu_snapshot.json"
SNAPSHOT_REQUEST = STATE_DIR / "gpu_snapshot.request"   # touched by viewers

ACTIVE   = ("queued", "running")
FINISHED = ("done", "failed", "killed", "cancelled", "lost")
MAX_RETRIES = 3


class TaskError(Exception):
    """A user-facing error: unknown task, bad arguments, ..."""

# ── Config ────────────────────────────────────────────────────────────────────

_CFG: Optional[Config] = None
_TRANSPORT: Optional[Transport] = None


def get_cfg() -> Config:
    """The loaded host config (raises ConfigError if the file is invalid)."""
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG


def get_transport() -> Transport:
    global _TRANSPORT
    if _TRANSPORT is None:
        _TRANSPORT = Transport(get_cfg())
    return _TRANSPORT


def set_config(cfg: Config) -> None:
    """Replace the active config (the daemon does this when the file changes)."""
    global _CFG, _TRANSPORT
    _CFG, _TRANSPORT = cfg, Transport(cfg)

# ── Database ──────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    command      TEXT    NOT NULL,
    num_gpus     INTEGER NOT NULL DEFAULT 1,
    min_mem      INTEGER NOT NULL DEFAULT 0,
    status       TEXT    NOT NULL DEFAULT 'queued',
    server       TEXT,
    gpu_indices  TEXT,
    ssh_pid      INTEGER,
    submitted_at REAL    NOT NULL,
    started_at   REAL,
    finished_at  REAL,
    exit_code    INTEGER,
    log_file     TEXT,
    working_dir  TEXT    NOT NULL,
    name         TEXT,
    retry_count  INTEGER NOT NULL DEFAULT 0
);
"""

# Columns added after the original schema: (name, definition).
_MIGRATIONS = [
    ("retry_count", "INTEGER NOT NULL DEFAULT 0"),
    ("min_mem",     "INTEGER NOT NULL DEFAULT 0"),
    ("job_key",     "TEXT"),
    ("remote_pid",  "INTEGER"),
    ("log_offset",  "INTEGER NOT NULL DEFAULT 0"),
    ("host_spec",   "TEXT"),
    ("remote_dir",  "TEXT"),
    ("last_seen",   "REAL"),
    ("repo_root",   "TEXT"),   # local git repo the task was submitted from
    ("gpu_type",    "TEXT"),   # --gpu-type: nvidia-smi model name(s), comma separated
]

# Fields reset when a task goes back to the queue for another attempt.
REQUEUE = ("status='queued', ssh_pid=NULL, started_at=NULL, finished_at=NULL, "
           "server=NULL, gpu_indices=NULL, job_key=NULL, remote_pid=NULL, "
           "log_offset=0, remote_dir=NULL, retry_count=retry_count+1")


def db_open() -> sqlite3.Connection:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")

    # Run schema creation
    con.executescript(_SCHEMA)

    # Manual migration for missing columns
    cursor = con.execute("PRAGMA table_info(tasks)")
    cols = {row[1] for row in cursor.fetchall()}
    for name, definition in _MIGRATIONS:
        if name not in cols:
            con.execute(f"ALTER TABLE tasks ADD COLUMN {name} {definition}")
            con.commit()

    return con

# ── Daemon process management ─────────────────────────────────────────────────

def daemon_pid() -> Optional[int]:
    """PID of the running daemon, or None."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)   # signal 0 = check existence only
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def daemon_is_running() -> bool:
    return daemon_pid() is not None


def start_daemon() -> Optional[int]:
    """Start the daemon; returns its PID, or None if it was already running."""
    if daemon_is_running():
        return None

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    pkg_root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.Popen(
        [sys.executable, "-m", "gpu_spooler.daemon"],
        stdout=open(DAEMON_LOG, "a"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,   # detach from terminal
        close_fds=True,
        cwd=str(Path.home()),
        env=env,
    )
    PID_FILE.write_text(str(proc.pid))
    return proc.pid


def stop_daemon() -> Optional[int]:
    """Stop the daemon; returns the PID it had, or None if it wasn't running."""
    pid = daemon_pid()
    PID_FILE.unlink(missing_ok=True)
    if pid is None:
        return None
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return None
    return pid

# ── Formatting shared by the front ends ───────────────────────────────────────

def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "-"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")


def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def one_line(s: str) -> str:
    """Collapse newlines/whitespace runs, for table cells."""
    return " ".join(s.split())


def runtime(row) -> Optional[float]:
    started = row["started_at"]
    ended = row["finished_at"] or (time.time() if row["status"] == "running" else None)
    return (ended - started) if started and ended else None

# ── Queries ───────────────────────────────────────────────────────────────────

def list_tasks(limit: Optional[int] = 200, con: Optional[sqlite3.Connection] = None) -> list:
    """Most recent tasks first."""
    con = con or db_open()
    query = "SELECT * FROM tasks ORDER BY id DESC" + ("" if limit is None else f" LIMIT {int(limit)}")
    return con.execute(query).fetchall()


def task_counts(con: Optional[sqlite3.Connection] = None) -> dict[str, int]:
    con = con or db_open()
    counts = {"total": 0, "running": 0, "queued": 0}
    for status, n in con.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status"):
        counts["total"] += n
        counts[status] = n
    return counts


def get_task(task_id: int, con: Optional[sqlite3.Connection] = None) -> sqlite3.Row:
    con = con or db_open()
    r = con.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if not r:
        raise TaskError(f"Task {task_id} not found.")
    return r


# Exit codes for `task wait` (see wait_exit_code).
EXIT_KILLED  = 130
EXIT_LOST    = 75
EXIT_TIMEOUT = 124
WAIT_POLL    = 2.0


def wait_exit_code(r) -> int:
    """How a finished task maps to a process exit code."""
    status = r["status"]
    if status == "done":
        return 0
    if status == "failed":
        return r["exit_code"] or 1
    if status in ("killed", "cancelled"):
        return EXIT_KILLED
    return EXIT_LOST


def describe_end(r) -> str:
    """e.g. 'task 12: done (exit 0, 3m 41s)'."""
    bits = []
    if r["status"] in ("done", "failed") and r["exit_code"] is not None:
        bits.append(f"exit {r['exit_code']}")
    if r["started_at"]:
        bits.append(fmt_duration(runtime(r)))
    return f"task {r['id']}: {r['status']}" + (f" ({', '.join(bits)})" if bits else "")


def wait_for(task_ids: list[int], timeout: Optional[float] = None,
             poll: float = WAIT_POLL) -> tuple[dict, list[int]]:
    """
    Block until every task is in a final state, reading only the database
    (the daemon keeps it current). Returns ({id: final row}, [ids still
    unfinished when the timeout hit]). Raises TaskError for an unknown id.
    """
    con = db_open()
    for tid in task_ids:
        get_task(tid, con)
    deadline = None if timeout is None else time.monotonic() + timeout
    done: dict = {}
    while True:
        for tid in task_ids:
            if tid not in done:
                r = get_task(tid, con)
                if r["status"] in FINISHED:
                    done[tid] = r
        pending = [t for t in task_ids if t not in done]
        if not pending:
            return done, []
        if deadline is not None:
            left = deadline - time.monotonic()
            if left <= 0:
                return done, pending
            time.sleep(min(poll, left))
        else:
            time.sleep(poll)


def task_details(r) -> list[tuple[str, str]]:
    """(label, value) rows describing a task, for `task info` and the TUI."""
    out = [("Status", r["status"])]
    if r["name"]:
        out.append(("Label", r["name"]))
    out.append(("Command", r["command"]))
    out.append(("GPUs req", str(r["num_gpus"] or "none (CPU job)")))
    if r["gpu_type"]:
        out.append(("GPU type", r["gpu_type"]))
    if r["min_mem"]:
        out.append(("Min VRAM", f"{r['min_mem']} GB"))
    if r["host_spec"]:
        out.append(("Hosts", r["host_spec"]))
    out.append(("Working dir", r["working_dir"]))
    if r["repo_root"]:
        out.append(("Git repo", r["repo_root"]))
    if r["remote_dir"] and r["remote_dir"] != r["working_dir"]:
        out.append(("Remote dir", r["remote_dir"]))
    out.append(("Submitted", fmt_ts(r["submitted_at"])))
    if r["server"]:
        server = r["server"]
        try:
            host = get_cfg().hosts.get(server)
        except ConfigError:
            host = None
        models = snapshot_models(server, r["gpu_indices"])
        if models:
            server += f"  ({', '.join(models)})"
        elif host and (host.gpu_type or host.gpu_mem_gb):
            server += f"  ({host.gpu_type or ''}  {host.gpu_mem_gb or '?'} GB/GPU, configured)"
        out.append(("Server", server))
    if r["gpu_indices"]:
        out.append(("GPU indices", r["gpu_indices"]))
    if r["job_key"]:
        out.append(("Job", f"~/.task-spooler/jobs/{r['job_key']}"
                    + (f"  (PGID {r['remote_pid']})" if r["remote_pid"] else "")))
    if r["status"] == "running" and r["last_seen"]:
        out.append(("Last seen", f"{fmt_duration(time.time() - r['last_seen'])} ago"))
    if r["started_at"]:
        out.append(("Started", fmt_ts(r["started_at"])))
    if r["finished_at"]:
        out.append(("Finished", fmt_ts(r["finished_at"])))
    if r["started_at"]:
        out.append(("Runtime", fmt_duration(runtime(r))))
    if r["exit_code"] is not None:
        out.append(("Exit code", str(r["exit_code"])))
    if r["retry_count"]:
        out.append(("Retries", f"{r['retry_count']}/{MAX_RETRIES}"))
    if r["log_file"]:
        out.append(("Log file", r["log_file"]))
    return out


def log_path(r) -> Optional[Path]:
    return Path(r["log_file"]) if r["log_file"] else None


def read_log_tail(path: Path, max_bytes: int = 256 * 1024) -> tuple[str, int]:
    """Last max_bytes of a log, and the file size (to continue from)."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        data = f.read()
    text = data.decode(errors="replace")
    if size > max_bytes:
        text = text.split("\n", 1)[-1]          # drop the partial first line
    return text, size


def read_log_from(path: Path, offset: int) -> tuple[str, int]:
    """Log bytes appended since offset, and the new offset."""
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    return data.decode(errors="replace"), offset + len(data)

# ── Submitting ────────────────────────────────────────────────────────────────

@dataclass
class Submitted:
    task_id: int
    command: str
    working_dir: str
    num_gpus: int
    min_mem: int
    name: Optional[str]
    host_spec: Optional[str]
    candidates: list[Host]
    warnings: list[str] = field(default_factory=list)
    gpu_type: Optional[str] = None


def command_from_argv(argv: list[str]) -> str:
    """
    A single argument is taken as shell code (task submit 'a && b'); several
    arguments are quoted so they reach the command unchanged.
    """
    return argv[0] if len(argv) == 1 else shlex.join(argv)


def on_ecs_filesystem(path: str) -> bool:
    """Best guess at whether `path` is visible as-is on the ECS servers."""
    if socket.gethostname().endswith(".vuw.ac.nz"):
        return True
    return path.startswith(("/vol/", "/am/")) or (
        sys.platform.startswith("linux") and Path("/vol").is_dir())


def logical_cwd() -> str:
    """The cwd as the shell spells it ($PWD keeps symlinks, like `pwd -L`)."""
    cwd = os.getcwd()
    pwd = os.environ.get("PWD")
    try:
        if pwd and os.path.isabs(pwd) and os.path.samefile(pwd, cwd):
            return os.path.normpath(pwd)
    except OSError:
        pass
    return cwd


def submit(command: str, num_gpus: int, min_mem: int = 0, name: Optional[str] = None,
           host_spec: Optional[str] = None, working_dir: Optional[str] = None,
           start: bool = True, check_repo: bool = True,
           gpu_type: Optional[str] = None) -> Submitted:
    """
    Queue a task (and start the daemon unless start=False). When submitted
    from a git repo, the remote copies are checked and any that are missing or
    stale produce a warning (check_repo=False skips the ssh round trip).
    gpu_type restricts the task to GPUs whose nvidia-smi name is one of these
    models (comma separated; see gpu.gpu_type_matches).
    """
    if num_gpus < 0:
        raise TaskError("-G must be 0 (CPU job) or more")
    if not command.strip():
        raise TaskError("empty command")
    gpu_type = ",".join(t.strip() for t in (gpu_type or "").split(",") if t.strip()) or None
    if gpu_type and num_gpus == 0:
        raise TaskError("--gpu-type needs a GPU job (-G 1 or more)")
    cfg = get_cfg()
    if cfg.max_gpus is not None and num_gpus > cfg.max_gpus:
        raise TaskError(f"-G {num_gpus} is more than [limits] max_gpus = {cfg.max_gpus}; "
                        "the task could never start")
    try:
        candidates = cfg.select(host_spec)
    except ConfigError as e:
        raise TaskError(f"-H {host_spec}: {e}") from None
    working_dir = os.path.abspath(os.path.expanduser(working_dir)) if working_dir else logical_cwd()

    # Off an ECS filesystem, a git repo is run from its remote copy (see repo.py).
    local_repo = None if on_ecs_filesystem(working_dir) else repo.find_repo(working_dir)

    con = db_open()
    cur = con.execute(
        "INSERT INTO tasks (command, num_gpus, min_mem, submitted_at, working_dir, name, "
        "host_spec, repo_root, gpu_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (command, num_gpus, min_mem or 0, time.time(), working_dir, name, host_spec or None,
         local_repo.root if local_repo else None, gpu_type),
    )
    con.commit()
    sub = Submitted(cur.lastrowid, command, working_dir, num_gpus, min_mem or 0, name,
                    host_spec or None, candidates, gpu_type=gpu_type)
    if gpu_type:
        problem = gpu_type_problem(gpu_type, candidates)
        if problem:
            sub.warnings.append(problem)

    if num_gpus == 0 and not any(h.cpu_slots > 0 for h in candidates):
        sub.warnings.append("no selected host has cpu_slots > 0; this task will stay "
                            "queued (see `task hosts`).")
    copies = repo.targets(cfg, candidates, local_repo, working_dir) if local_repo else []
    covered = {name for t in copies for name in t.members}
    unmapped = [h.name for h in candidates if not h.is_local and h.name not in covered
                and not cfg.is_mapped(h, working_dir)]
    if unmapped and not on_ecs_filesystem(working_dir):
        sub.warnings.append(f"{working_dir} has no path_map entry for "
                            f"{truncate(', '.join(unmapped), 40)}; it will be used unchanged there.")
    if check_repo and copies:
        sub.warnings += [w for _, w in repo_status(local_repo, copies) if w]
    if start:
        start_daemon()
    return sub


def repo_status(local_repo: "repo.LocalRepo", copies: list) -> list[tuple["repo.Target", Optional[str]]]:
    """(target, problem or None) for each remote copy, checked in parallel."""
    transport = get_transport()
    by_host = {t.host.name: t for t in copies}
    states = transport.fan_out([t.host for t in copies],
                               lambda h: repo.check(transport, by_host[h.name]))
    return [(t, repo.staleness(local_repo, t, states[t.host.name])) for t in copies]


@dataclass
class RepoInfo:
    repo: "repo.LocalRepo"
    copies: list          # repo.Target
    working_dir: str


def repo_for(working_dir: Optional[str] = None, host_spec: Optional[str] = None) -> RepoInfo:
    """The local git repo at working_dir and where its remote copies live."""
    working_dir = os.path.abspath(os.path.expanduser(working_dir)) if working_dir else logical_cwd()
    local_repo = repo.find_repo(working_dir)
    if local_repo is None:
        raise TaskError(f"{working_dir} is not inside a git repo")
    cfg = get_cfg()
    try:
        hosts = cfg.select(host_spec)
    except ConfigError as e:
        raise TaskError(f"-H {host_spec}: {e}") from None
    copies = repo.targets(cfg, hosts, local_repo, working_dir)
    if not copies:
        raise TaskError("no selected host has remote_root set (or path_map already covers "
                        f"{working_dir}); see the Configuration section of the README")
    return RepoInfo(local_repo, copies, working_dir)


def sync_repo(info: RepoInfo) -> list["repo.SyncResult"]:
    """Push the local working tree to every remote copy (in parallel)."""
    transport = get_transport()
    by_host = {t.host.name: t for t in info.copies}
    results = transport.fan_out([t.host for t in info.copies],
                                lambda h: repo.sync(transport, info.repo, by_host[h.name]))
    return [results[t.host.name] for t in info.copies]


def running_in(paths: list[str]) -> int:
    """How many running tasks use a working directory under one of these remote paths."""
    rows = db_open().execute(
        "SELECT remote_dir FROM tasks WHERE status='running' AND remote_dir IS NOT NULL").fetchall()
    return sum(1 for r in rows
               if any(r[0] == p or r[0].startswith(p.rstrip("/") + "/") for p in paths))


def resubmit(task_id: int) -> Submitted:
    """Queue a copy of an existing task (same command, resources and directory)."""
    r = get_task(task_id)
    return submit(r["command"], r["num_gpus"], r["min_mem"], r["name"], r["host_spec"],
                  r["working_dir"], gpu_type=r["gpu_type"])

# ── Killing ───────────────────────────────────────────────────────────────────

@dataclass
class KillResult:
    task_id: int
    action: str          # cancelled | killed | none
    confirmed: bool      # False: the host could not be reached
    message: str


def _kill_remote(row) -> tuple[bool, str]:
    """
    Signal a running task's process group on its host.
    Returns (confirmed, message); confirmed is False if the host was unreachable.
    """
    if not row["job_key"]:
        # Started by the old version: the job lives inside a local ssh client.
        if row["ssh_pid"]:
            try:
                os.kill(row["ssh_pid"], signal.SIGTERM)
                return True, f"Sent SIGTERM to SSH client (PID {row['ssh_pid']})."
            except ProcessLookupError:
                return True, "SSH process already gone."
        return True, "No process recorded."
    host = get_cfg().hosts.get(row["server"])
    if host is None:
        return False, f"host '{row['server']}' is not in the config"
    rc, out, err = get_transport().run(host, jobs.kill_script(row["job_key"]), timeout=30)
    if rc == 0 and "KILLED" in out:
        return True, (f"Sent SIGTERM to process group {row['remote_pid'] or '?'} on "
                      f"{host.name} (SIGKILL follows in 10s if it is still alive).")
    if rc == 0 and ("NOT_RUNNING" in out or "NO_JOB" in out):
        return True, f"Process on {host.name} was already gone."
    reason = (err.strip().splitlines() or [f"exit {rc}"])[-1]
    return False, (f"could not reach {host.name}: {reason}. The process may still be "
                   f"running; check with `task ssh {row['server']}`"
                   + (f" and `kill -- -{row['remote_pid']}`." if row["remote_pid"] else "."))


def kill(task_id: int) -> KillResult:
    """Cancel a queued task or kill a running one."""
    con = db_open()
    r = get_task(task_id, con)
    status = r["status"]
    if status == "queued":
        n = con.execute(
            "UPDATE tasks SET status='cancelled', finished_at=? WHERE id=? AND status='queued'",
            (time.time(), task_id),
        ).rowcount
        con.commit()
        if n:
            return KillResult(task_id, "cancelled", True, f"Task {task_id} cancelled.")
        r = get_task(task_id, con)       # it started in the meantime
        status = r["status"]
    if status != "running":
        return KillResult(task_id, "none", True, f"Task {task_id} is already {status}.")

    # Mark killed first so the daemon doesn't treat the signal as a failure
    # and retry it.
    con.execute(
        "UPDATE tasks SET status='killed', finished_at=?, ssh_pid=NULL WHERE id=?",
        (time.time(), task_id),
    )
    con.commit()
    ok, msg = _kill_remote(r)
    return KillResult(task_id, "killed", ok, msg)


@dataclass
class KillAllResult:
    cancelled: int
    killed: int
    unconfirmed: list[KillResult]


def kill_all() -> KillAllResult:
    """Cancel every queued task and kill every running one, in parallel."""
    con = db_open()
    now = time.time()
    n_queued = con.execute(
        "UPDATE tasks SET status='cancelled', finished_at=? WHERE status='queued'",
        (now,),
    ).rowcount
    running = con.execute("SELECT * FROM tasks WHERE status='running'").fetchall()
    n_running = con.execute(
        "UPDATE tasks SET status='killed', finished_at=?, ssh_pid=NULL WHERE status='running'",
        (now,),
    ).rowcount
    con.commit()

    cfg = get_cfg()
    get_transport().warm_up([cfg.hosts[r["server"]] for r in running if r["server"] in cfg.hosts])
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(_kill_remote, running))
    unconfirmed = [KillResult(r["id"], "killed", False, msg)
                   for r, (ok, msg) in zip(running, results) if not ok]
    return KillAllResult(n_queued, n_running, unconfirmed)


def reset_history() -> None:
    """Delete every task and restart IDs at 1."""
    con = db_open()
    con.execute("DELETE FROM tasks")
    con.execute("DELETE FROM sqlite_sequence WHERE name='tasks'")
    con.commit()

# ── Clearing ──────────────────────────────────────────────────────────────────

@dataclass
class Cleared:
    tasks: int
    log_files: int
    failed_hosts: list[str]


def clear_finished() -> Cleared:
    """Remove finished tasks, their local logs and their job directories."""
    con = db_open()
    finished = f"status IN ({','.join(repr(s) for s in FINISHED)})"
    to_delete = con.execute(f"SELECT id, server, log_file FROM tasks WHERE {finished}").fetchall()
    n = con.execute(f"DELETE FROM tasks WHERE {finished}").rowcount
    con.commit()

    deleted_files = 0
    for r in to_delete:
        if r["log_file"]:
            try:
                Path(r["log_file"]).unlink()
                deleted_files += 1
            except OSError:
                pass

    # Remove job directories on the hosts (best effort).
    cfg = get_cfg()
    by_host: dict[str, list[int]] = {}
    for r in to_delete:
        if r["server"] in cfg.hosts:
            by_host.setdefault(r["server"], []).append(r["id"])
    transport = get_transport()
    cleaned = transport.fan_out(
        [cfg.hosts[h] for h in by_host],
        lambda h: transport.run(h, jobs.cleanup_script(by_host[h.name]), timeout=30)[0] == 0,
    )
    return Cleared(n, deleted_files, sorted(h for h, ok in cleaned.items() if not ok))

# ── GPU snapshot (written by the daemon) ──────────────────────────────────────

def read_gpu_snapshot() -> Optional[dict]:
    """The daemon's latest GPU query: {"ts", "mode", "hosts": {name: ServerResult dict}}."""
    try:
        return json.loads(SNAPSHOT_PATH.read_text())
    except (OSError, ValueError):
        return None


def snapshot_models(host_name: str, gpu_indices: Optional[str] = None) -> list[str]:
    """GPU models a host reported in the latest snapshot (only these indices, if given)."""
    from gpu_spooler.gpu import gpu_model
    h = ((read_gpu_snapshot() or {}).get("hosts") or {}).get(host_name) or {}
    wanted = {int(i) for i in (gpu_indices or "").split(",") if i.strip().isdigit()}
    names = [g.get("name", "") for g in h.get("gpus", [])
             if not wanted or g.get("index") in wanted]
    return list(dict.fromkeys(gpu_model(n) for n in names if n))


def gpu_type_problem(gpu_type: str, candidates: list[Host]) -> Optional[str]:
    """
    A warning if no candidate host reported a GPU of this type in the latest
    snapshot (None if one did, or if there is no snapshot to judge by).
    """
    from gpu_spooler.gpu import gpu_type_matches
    hosts = (read_gpu_snapshot() or {}).get("hosts") or {}
    seen = {n: snapshot_models(n) for n in (h.name for h in candidates) if n in hosts}
    if not any(seen.values()):
        return None
    if any(gpu_type_matches(gpu_type, m) for models in seen.values() for m in models):
        return None
    known = sorted({m for models in seen.values() for m in models})
    return (f"no selected host reported a GPU of type '{gpu_type}' in the last GPU poll "
            f"(seen: {', '.join(known)}); the task will stay queued unless one does.")


def request_gpu_snapshot() -> None:
    """Ask the daemon to keep the GPU snapshot fresh for the next couple of minutes."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_REQUEST.touch()
