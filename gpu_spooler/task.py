#!/usr/bin/env python3
"""
task - GPU task spooler for SSH-reachable GPU servers (VUW ECS by default)

The background daemon auto-starts on first job submission and dispatches
queued tasks to GPU servers as GPUs become free. Jobs run detached on the
server, so they survive the SSH connection (or this machine) going away.

Submission:
  task -G 1 python3 train.py             queue a job needing 1 GPU
  task -G 2 -n "experiment A" ./run.sh   queue with a label
  task -G 1 -H cuda14,cuda15 ./run.sh    only run on these hosts / groups / tags
  task -G 0 ./preprocess.sh              CPU-only job (hosts with cpu_slots)

Inspection:
  task -l  / --list                      list tasks (most recent 200)
  task -L  / --all                       list all tasks (no limit)
  task -i 3 / --info 3                   detailed info for task 3
  task -o 3 / --output 3                 print saved output of task 3
  task -f 3 / --follow 3                 stream live output of task 3

Management:
  task -k 3 / --kill 3                   kill / cancel task 3
  task -K   / --kill-all                 kill all running & queued tasks, stop daemon
  task -c   / --clear                    remove finished tasks from the list
  task -w 3 / --wait 3                   block until task 3 finishes

Hosts / daemon:
  task --gpu-status                      show GPU availability across all servers
  task --hosts                           show configured hosts and network mode
  task --connect                         log in to jump hosts once (password/OTP)
  task --ssh cuda14                      interactive shell on a host
  task --init-config                     write ~/.task-spooler/hosts.toml template
  task --stop                            stop the background daemon
  task --daemon                          (re)start the daemon explicitly
"""

import argparse
import concurrent.futures
import os
import shlex
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from gpu_spooler import jobs
from gpu_spooler.config import (
    STATE_DIR, Config, ConfigError, Host, config_path, config_template, load_config,
)
from gpu_spooler.gpu import ServerResult, query_hosts, write_snapshot
from gpu_spooler.transport import Transport, auth_hint

# ── Paths ─────────────────────────────────────────────────────────────────────

DB_PATH       = STATE_DIR / "tasks.db"
PID_FILE      = STATE_DIR / "daemon.pid"
DAEMON_LOG    = STATE_DIR / "daemon.log"
LOGS_DIR      = STATE_DIR / "logs"
SNAPSHOT_PATH = STATE_DIR / "gpu_snapshot.json"

POLL_INTERVAL   = 30   # seconds between GPU polls / dispatch in the daemon
STATUS_INTERVAL = float(os.environ.get("TASK_SPOOLER_STATUS_INTERVAL", 5))
                       # seconds between status / log-sync polls of running jobs
LAUNCH_TIMEOUT  = 60
STATUS_TIMEOUT  = 60

# ── Config ────────────────────────────────────────────────────────────────────

_CFG: Optional[Config] = None
_TRANSPORT: Optional[Transport] = None


def get_cfg() -> Config:
    global _CFG
    if _CFG is None:
        try:
            _CFG = load_config()
        except ConfigError as e:
            sys.exit(f"task: config error: {e}")
    return _CFG


def get_transport() -> Transport:
    global _TRANSPORT
    if _TRANSPORT is None:
        _TRANSPORT = Transport(get_cfg())
    return _TRANSPORT

# ── Assignment ────────────────────────────────────────────────────────────────


@dataclass
class HostState:
    """What a host can take right now, for one dispatch round."""
    host: Host
    free: dict[int, bool]          # gpu index → free
    cpu_free: int                  # unused CPU slots
    detected_mem_gb: float = 0.0   # smallest detected GPU memory (decimal GB)

    @property
    def mem_gb(self) -> float:
        return self.host.gpu_mem_gb if self.host.gpu_mem_gb is not None else self.detected_mem_gb


def host_states(cfg: Config, results: dict[str, ServerResult], running_rows) -> dict[str, HostState]:
    """Build per-host availability, treating our own running tasks as busy."""
    states: dict[str, HostState] = {}
    for name, r in results.items():
        if not r.reachable or name not in cfg.hosts:
            continue
        states[name] = HostState(
            host=cfg.hosts[name],
            free={g.index: g.is_free for g in r.gpus},
            cpu_free=cfg.hosts[name].cpu_slots,
            detected_mem_gb=min((g.mem_total_gb for g in r.gpus), default=0.0),
        )
    for row in running_rows:
        st = states.get(row["server"])
        if st is None:
            continue
        if row["num_gpus"] == 0:
            st.cpu_free -= 1
        for idx in (row["gpu_indices"] or "").split(","):
            if idx.strip().isdigit() and int(idx) in st.free:
                st.free[int(idx)] = False
    return states


def assign_tasks(task_rows, states: dict[str, HostState], cfg: Config) -> list[tuple]:
    """
    Greedily assign tasks (in queue order) to hosts. GPU tasks bin-pack onto
    hosts with the fewest GPUs first; -G 0 tasks use free CPU slots.
    Mutates states to reserve what is assigned.
    Returns list of (task_id, host_name, [gpu_indices]).
    """
    order = sorted(states.values(), key=lambda s: len(s.free))
    assignments = []
    for row in task_rows:
        n = row["num_gpus"]
        min_mem = row["min_mem"] or 0
        try:
            allowed = {h.name for h in cfg.select(row["host_spec"])}
        except ConfigError:
            continue
        for st in order:
            if st.host.name not in allowed:
                continue
            if n == 0:
                if st.cpu_free > 0:
                    st.cpu_free -= 1
                    assignments.append((row["id"], st.host.name, []))
                    break
                continue
            if st.mem_gb < min_mem:
                continue
            free = [i for i, ok in sorted(st.free.items()) if ok]
            if len(free) >= n:
                chosen = free[:n]
                for idx in chosen:
                    st.free[idx] = False   # reserve
                assignments.append((row["id"], st.host.name, chosen))
                break
    return assignments

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
]

MAX_RETRIES = 3

# Fields reset when a task goes back to the queue for another attempt.
_REQUEUE = ("status='queued', ssh_pid=NULL, started_at=NULL, finished_at=NULL, "
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

# ── Daemon ────────────────────────────────────────────────────────────────────

class Daemon:
    """
    Background process that dispatches queued tasks to hosts and polls the
    detached jobs for completion, syncing their logs. All job state lives in
    the database and the job directories, so a restarted daemon re-adopts
    jobs that are still running.
    """

    def __init__(self):
        self.con = db_open()
        self.cfg = get_cfg()
        self.transport = get_transport()
        self._queued_seen: set[int] = set()
        self._last_dispatch = 0.0
        self._last_status = 0.0
        self._dispatch_due = True
        self._status_due = True
        self._mode: Optional[str] = None
        self._unknown_hosts: set[str] = set()

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    def run(self) -> None:
        self._log(f"Daemon started. Config: {self.cfg.source_label}")
        # Tasks started by the old (non-detached) version cannot be re-adopted.
        n = self.con.execute(
            "UPDATE tasks SET status='lost', finished_at=? "
            "WHERE status='running' AND job_key IS NULL",
            (time.time(),),
        ).rowcount
        if n:
            self._log(f"Marked {n} task(s) from the previous version as 'lost'.")
        self.con.commit()
        n = self.con.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='running'").fetchone()[0]
        if n:
            self._log(f"Re-adopting {n} running task(s).")

        while True:
            try:
                self._tick()
            except Exception as e:
                self._log(f"Error in daemon loop: {e!r}")
                time.sleep(STATUS_INTERVAL)
            time.sleep(1)

    def _tick(self) -> None:
        self._maybe_reload_config()
        now = time.time()

        mode = self.transport.mode()
        if mode != self._mode:
            self._log(f"Network mode: {self.transport.describe_mode()}")
            self._mode = mode

        if self._status_due or now - self._last_status >= STATUS_INTERVAL:
            self._status_due = False
            self._last_status = now
            self._poll_running()

        queued = {r[0] for r in self.con.execute(
            "SELECT id FROM tasks WHERE status='queued'").fetchall()}
        if queued - self._queued_seen:
            self._dispatch_due = True
        self._queued_seen = queued
        if queued and (self._dispatch_due or now - self._last_dispatch >= POLL_INTERVAL):
            self._dispatch_due = False
            self._last_dispatch = time.time()
            self._dispatch()

    def _maybe_reload_config(self) -> None:
        path = config_path()
        try:
            mtime = path.stat().st_mtime if path.exists() else 0.0
        except OSError:
            return
        current = self.cfg.mtime if self.cfg.source else 0.0
        if mtime == current:
            return
        global _CFG, _TRANSPORT
        try:
            cfg = load_config()
        except ConfigError as e:
            self._log(f"Config reload failed, keeping previous config: {e}")
            self.cfg.mtime = mtime          # don't retry until it changes again
            return
        _CFG, _TRANSPORT = cfg, Transport(cfg)
        self.cfg, self.transport = _CFG, _TRANSPORT
        self._unknown_hosts.clear()
        self._log(f"Config reloaded from {cfg.source_label}.")

    # ── status / log sync ────────────────────────────────────────────────────

    def _poll_running(self) -> None:
        rows = self.con.execute(
            "SELECT * FROM tasks WHERE status='running' AND job_key IS NOT NULL"
        ).fetchall()
        if not rows:
            return
        by_host: dict[str, list] = {}
        for r in rows:
            if r["server"] not in self.cfg.hosts:
                if r["server"] not in self._unknown_hosts:
                    self._log(f"Task {r['id']}: host '{r['server']}' is no longer "
                              "configured; cannot poll it.")
                    self._unknown_hosts.add(r["server"])
                continue
            by_host.setdefault(r["server"], []).append(r)

        def poll(host: Host):
            specs = [(r["job_key"], r["log_offset"] or 0) for r in by_host[host.name]]
            rc, out, err = self.transport.run(host, jobs.status_script(specs),
                                              timeout=STATUS_TIMEOUT)
            return jobs.parse_status(out) if rc == 0 else None

        results = self.transport.fan_out(
            [self.cfg.hosts[h] for h in by_host], poll)
        now = time.time()
        for host_name, statuses in results.items():
            if statuses is None:
                continue      # unreachable: leave tasks running, last_seen unchanged
            for row in by_host[host_name]:
                st = statuses.get(row["job_key"])
                if st is not None:
                    self._apply_status(row, st, now)

    def _apply_status(self, row, st: jobs.JobStatus, now: float) -> None:
        tid, key = row["id"], row["job_key"]
        offset = row["log_offset"] or 0
        if st.data:
            try:
                with open(row["log_file"], "ab") as f:
                    f.write(st.data)
                offset += len(st.data)
            except (OSError, TypeError) as e:
                self._log(f"Task {tid}: cannot append to log: {e}")
        self.con.execute(
            "UPDATE tasks SET log_offset=?, last_seen=? WHERE id=? AND job_key=?",
            (offset, now, tid, key))
        self.con.commit()

        if st.state == "running":
            return
        if offset < st.size:
            self._status_due = True      # fetch the rest of the log right away
            return
        if st.state == "done":
            self._finish(row, st.rc)
        else:
            self._update_running(
                tid, key, "status='lost', exit_code=NULL, finished_at=?", (now,))
            self._log(f"Task {tid} lost on {row['server']} "
                      f"(job directory {st.state}; not retrying).")
        self._dispatch_due = True        # GPUs/slots were freed

    def _update_running(self, tid: int, key: str, assignments: str, params=()) -> bool:
        """UPDATE a task only if it is still this running attempt."""
        n = self.con.execute(
            f"UPDATE tasks SET {assignments} "
            "WHERE id=? AND job_key=? AND status='running'",
            (*params, tid, key)).rowcount
        self.con.commit()
        return n > 0

    def _finish(self, row, rc: Optional[int]) -> None:
        tid, key = row["id"], row["job_key"]
        if rc == 0:
            if self._update_running(tid, key, "status='done', exit_code=0, finished_at=?",
                                    (time.time(),)):
                self._log(f"Task {tid} finished — exit 0 (done).")
            return
        if (row["retry_count"] or 0) < MAX_RETRIES:
            if self._update_running(tid, key, f"{_REQUEUE}, exit_code=?", (rc,)):
                self._log(f"Task {tid} failed (exit {rc}), re-queuing "
                          f"(attempt {row['retry_count'] + 1}/{MAX_RETRIES}).")
        elif self._update_running(tid, key, "status='failed', exit_code=?, finished_at=?",
                                  (rc, time.time())):
            self._log(f"Task {tid} failed (exit {rc}), "
                      f"no more retries ({MAX_RETRIES}/{MAX_RETRIES}).")

    # ── dispatch ─────────────────────────────────────────────────────────────

    def _dispatch(self) -> None:
        """Query hosts and start queued tasks on free GPUs / CPU slots."""
        rows = self.con.execute(
            "SELECT * FROM tasks WHERE status='queued' ORDER BY id"
        ).fetchall()
        if not rows:
            return

        candidates: dict[str, Host] = {}
        for r in rows:
            try:
                for h in self.cfg.select(r["host_spec"]):
                    candidates[h.name] = h
            except ConfigError as e:
                self._log(f"Task {r['id']}: -H {r['host_spec']!r}: {e}; it stays queued.")

        self._log(f"Polling {len(candidates)} host(s) for {len(rows)} queued task(s)...")
        results = query_hosts(self.transport, candidates.values())
        try:
            write_snapshot(SNAPSHOT_PATH, results, self.transport.mode())
        except OSError:
            pass

        running = self.con.execute(
            "SELECT server, gpu_indices, num_gpus FROM tasks WHERE status='running'"
        ).fetchall()
        states = host_states(self.cfg, results, running)
        if not states:
            hints = {auth_hint(r.error) for r in results.values()} - {None}
            self._log("No reachable hosts (check SSH keys/agent"
                      + ("; run `task --connect`" if hints else "") + ").")
            return

        avail = ", ".join(
            f"{n}:{sum(s.free.values())}" + (f"+{s.cpu_free}cpu" if s.cpu_free > 0 else "")
            for n, s in sorted(states.items()))
        self._log(f"Free: {avail}")

        assignments = assign_tasks(rows, states, self.cfg)
        if not assignments:
            self._log("No free slots available; tasks remain queued.")
            return

        for task_id, host_name, gpu_indices in assignments:
            self._launch(task_id, self.cfg.hosts[host_name], gpu_indices)

    def _launch(self, task_id: int, host: Host, gpu_indices: list[int]) -> None:
        row = self.con.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            return

        gpu_str  = ",".join(str(i) for i in gpu_indices)
        log_path = LOGS_DIR / f"{task_id}.log"
        key      = jobs.new_job_key(task_id)
        wdir     = row["working_dir"]
        rdir     = wdir if host.is_local else self.cfg.map_path(host, wdir)

        # Reserve in DB before launching, so a daemon restart mid-launch can
        # find (or give up on) the job directory.
        n = self.con.execute(
            "UPDATE tasks SET status='running', server=?, gpu_indices=?, started_at=?, "
            "log_file=?, job_key=?, remote_pid=NULL, log_offset=0, remote_dir=?, "
            "last_seen=NULL, finished_at=NULL WHERE id=? AND status='queued'",
            (host.name, gpu_str, time.time(), str(log_path), key, rdir, task_id),
        ).rowcount
        self.con.commit()
        if not n:
            return      # cancelled in the meantime

        # Write a header to the log
        with open(log_path, "w") as f:
            f.write(
                f"# task {task_id}: {row['command']}\n"
                f"# server: {host.name}  CUDA_VISIBLE_DEVICES: {gpu_str or '(none)'}\n"
                f"# started: {datetime.now().isoformat()}\n"
                f"# working_dir: {rdir}"
                + (f"  (mapped from {wdir})" if rdir != wdir else "") + "\n"
                f"{'─' * 60}\n"
            )

        # Multi-GPU jobs sleep briefly before starting to reduce the chance of
        # torchrun port collisions when several launch at the same time.
        script = jobs.launch_script(key, rdir, gpu_str, row["command"],
                                    delay=len(gpu_indices) > 1)
        rc, out, err = self.transport.run(host, script, timeout=LAUNCH_TIMEOUT)
        pid = jobs.parse_launch(out)
        if pid:
            self.con.execute("UPDATE tasks SET remote_pid=? WHERE id=? AND job_key=?",
                             (pid, task_id, key))
            self.con.commit()
            self._log(f"Task {task_id} dispatched → {host.name} "
                      f"[CUDA_VISIBLE_DEVICES={gpu_str}]  (job {key}, PGID {pid})")
            return
        if rc == 124:
            # The job may or may not have started; the status poll will tell.
            self._log(f"Task {task_id}: launch on {host.name} timed out; "
                      "will check its job directory.")
            return

        reason = (err.strip().splitlines() or [f"exit {rc}"])[-1]
        hint = auth_hint(err)
        with open(log_path, "a") as f:
            f.write(f"task-spooler: launch on {host.name} failed: {err.strip() or out.strip()}\n")
        retry_count = row["retry_count"] or 0
        if retry_count < MAX_RETRIES:
            self._update_running(task_id, key, _REQUEUE)
            self._log(f"Task {task_id} failed to launch on {host.name}: {reason}"
                      + (f" ({hint})" if hint else "")
                      + f" — re-queuing (attempt {retry_count + 1}/{MAX_RETRIES}).")
        else:
            self._update_running(task_id, key, "status='failed', finished_at=?", (time.time(),))
            self._log(f"Task {task_id} failed to launch on {host.name}: {reason} — "
                      f"no more retries ({MAX_RETRIES}/{MAX_RETRIES}).")

# ── Daemon process management ─────────────────────────────────────────────────

def daemon_is_running() -> bool:
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)   # signal 0 = check existence only
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


def start_daemon(quiet: bool = False) -> None:
    if daemon_is_running():
        if not quiet:
            print("Daemon is already running.")
        return

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    pkg_root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.Popen(
        [sys.executable, "-m", "gpu_spooler.task", "--_daemon-loop"],
        stdout=open(DAEMON_LOG, "a"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,   # detach from terminal
        close_fds=True,
        cwd=str(Path.home()),
        env=env,
    )
    PID_FILE.write_text(str(proc.pid))
    if not quiet:
        print(f"Daemon started (PID {proc.pid}). Logs: {DAEMON_LOG}")


def stop_daemon() -> None:
    if not daemon_is_running():
        print("Daemon is not running.")
        return
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        PID_FILE.unlink(missing_ok=True)
        print(f"Daemon (PID {pid}) stopped. Running jobs keep running and are "
              "re-adopted when it restarts.")
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        print("Daemon was not running (cleaned up stale PID file).")

# ── CLI helpers ───────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
ORANGE = "\033[33m"

STATUS_COLOR = {
    "queued":    DIM,
    "running":   GREEN,
    "done":      CYAN,
    "failed":    RED,
    "killed":    YELLOW,
    "cancelled": YELLOW,
    "lost":      ORANGE,
}


def _col(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "-"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def _fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"

# ── Commands ──────────────────────────────────────────────────────────────────

def _on_ecs_filesystem(path: str) -> bool:
    """Best guess at whether `path` is visible as-is on the ECS servers."""
    if socket.gethostname().endswith(".vuw.ac.nz"):
        return True
    return path.startswith(("/vol/", "/am/")) or (
        sys.platform.startswith("linux") and Path("/vol").is_dir())


def _logical_cwd() -> str:
    """The cwd as the shell spells it ($PWD keeps symlinks, like `pwd -L`)."""
    cwd = os.getcwd()
    pwd = os.environ.get("PWD")
    try:
        if pwd and os.path.isabs(pwd) and os.path.samefile(pwd, cwd):
            return os.path.normpath(pwd)
    except OSError:
        pass
    return cwd


def cmd_submit(args) -> None:
    if args.gpus < 0:
        sys.exit("task: -G must be 0 (CPU job) or more")
    cfg = get_cfg()
    try:
        candidates = cfg.select(args.host)
    except ConfigError as e:
        sys.exit(f"task: -H {args.host}: {e}")

    # A single argument is taken as shell code (task -G 1 'a && b'); several
    # arguments are quoted so they reach the command unchanged.
    command = args.command[0] if len(args.command) == 1 else shlex.join(args.command)
    working_dir = os.path.abspath(args.dir) if args.dir else _logical_cwd()
    con = db_open()
    cur = con.execute(
        "INSERT INTO tasks (command, num_gpus, min_mem, submitted_at, working_dir, name, host_spec) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (command, args.gpus, args.min_mem, time.time(), working_dir, args.name, args.host),
    )
    con.commit()
    task_id = cur.lastrowid
    label = f'  "{args.name}"' if args.name else ""
    print(f"Queued task {task_id}{label}: {command}")
    need = "CPU only" if args.gpus == 0 else f"{args.gpus} GPU(s) (min {args.min_mem or 0}GB VRAM)"
    print(f"  Requires: {need}   Working dir: {working_dir}")
    if args.host:
        print(f"  Hosts: {', '.join(h.name for h in candidates)}")

    if args.gpus == 0 and not any(h.cpu_slots > 0 for h in candidates):
        print(_col("  Warning: no selected host has cpu_slots > 0; this task will stay "
                   "queued (see `task --hosts`).", YELLOW))
    unmapped = [h.name for h in candidates
                if not h.is_local and not cfg.is_mapped(h, working_dir)]
    if unmapped and not _on_ecs_filesystem(working_dir):
        print(_col(f"  Warning: {working_dir} has no path_map entry for "
                   f"{_truncate(', '.join(unmapped), 40)}; it will be used unchanged there.",
                   YELLOW))

    start_daemon(quiet=True)

    daemon_status = (
        _col("running", GREEN) if daemon_is_running()
        else _col("NOT running — start with: task --daemon", RED)
    )
    print(f"  Daemon: {daemon_status}")


def cmd_list(args) -> None:
    con = db_open()
    limit = getattr(args, "all", False)
    query = "SELECT * FROM tasks ORDER BY id DESC" + ("" if limit else " LIMIT 200")
    rows = con.execute(query).fetchall()
    if not rows:
        print("No tasks.")
        return

    hdr = (
        f"{'ID':>4}  {'STATUS':10}  {'G':>1}  {'HOST':<10}  {'SERVER':<14}  "
        f"{'GPU':>4}  {'SUBMITTED':11}  {'RUNTIME':>8}  LABEL / COMMAND"
    )
    sep = "─" * max(len(hdr), 80)
    print(f"\n{BOLD}{hdr}{RESET}")
    print(sep)

    for r in reversed(rows):
        status  = r["status"]
        color   = STATUS_COLOR.get(status, "")
        started = r["started_at"]
        ended   = r["finished_at"] or (time.time() if status == "running" else None)
        runtime = _fmt_duration((ended - started) if started and ended else None)
        label   = _truncate(r["name"] or r["command"], 42)
        spec    = _truncate(r["host_spec"] or "-", 10)
        server  = r["server"] or "-"
        gpus    = r["gpu_indices"] or "-"

        print(
            f"{r['id']:>4}  {_col(f'{status:10}', color)}  {r['num_gpus']:>1}  "
            f"{spec:<10}  {server:<14}  {gpus:>4}  {_fmt_ts(r['submitted_at']):11}  "
            f"{runtime:>8}  {label}"
        )

    print(sep)
    shown = len(rows)
    total = con.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    running = con.execute("SELECT COUNT(*) FROM tasks WHERE status='running'").fetchone()[0]
    queued  = con.execute("SELECT COUNT(*) FROM tasks WHERE status='queued'").fetchone()[0]
    daemon_str = (
        _col("running", GREEN) if daemon_is_running()
        else _col("stopped", RED)
    )
    truncated = f"  {_col(f'(showing {shown} of {total})', DIM)}" if shown < total else ""
    print(
        f"\n{total} task(s){truncated}  "
        f"{_col(str(running), GREEN)} running  "
        f"{_col(str(queued), YELLOW)} queued  "
        f"daemon: {daemon_str}  "
        f"network: {get_transport().describe_mode(probe=False)}\n"
    )


def cmd_info(args) -> None:
    con = db_open()
    r = con.execute("SELECT * FROM tasks WHERE id=?", (args.id,)).fetchone()
    if not r:
        print(f"Task {args.id} not found.")
        sys.exit(1)

    status = r["status"]
    color  = STATUS_COLOR.get(status, "")
    started  = r["started_at"]
    finished = r["finished_at"]
    runtime  = _fmt_duration((finished or time.time()) - started if started else None)

    print(f"\n{BOLD}Task {r['id']}{RESET}")
    print(f"  Status:      {_col(status, color)}")
    if r["name"]:
        print(f"  Label:       {r['name']}")
    print(f"  Command:     {r['command']}")
    print(f"  GPUs req:    {r['num_gpus'] or 'none (CPU job)'}")
    if r["min_mem"]:
        print(f"  Min VRAM:    {r['min_mem']} GB")
    if r["host_spec"]:
        print(f"  Hosts:       {r['host_spec']}")
    print(f"  Working dir: {r['working_dir']}")
    if r["remote_dir"] and r["remote_dir"] != r["working_dir"]:
        print(f"  Remote dir:  {r['remote_dir']}")
    print(f"  Submitted:   {_fmt_ts(r['submitted_at'])}")
    if r["server"]:
        print(f"  Server:      {r['server']}")
        host = get_cfg().hosts.get(r["server"])
        if host and (host.gpu_type or host.gpu_mem_gb):
            print(f"               ({host.gpu_type or ''}  {host.gpu_mem_gb or '?'} GB/GPU)")
    if r["gpu_indices"]:
        print(f"  GPU indices: {r['gpu_indices']}")
    if r["job_key"]:
        print(f"  Job:         ~/.task-spooler/jobs/{r['job_key']}"
              + (f"  (PGID {r['remote_pid']})" if r["remote_pid"] else ""))
    if status == "running" and r["last_seen"]:
        print(f"  Last seen:   {_fmt_duration(time.time() - r['last_seen'])} ago")
    if started:
        print(f"  Started:     {_fmt_ts(started)}")
    if finished:
        print(f"  Finished:    {_fmt_ts(finished)}")
    if started:
        print(f"  Runtime:     {runtime}")
    if r["exit_code"] is not None:
        print(f"  Exit code:   {r['exit_code']}")
    if r["retry_count"]:
        print(f"  Retries:     {r['retry_count']}/{MAX_RETRIES}")
    if r["log_file"]:
        print(f"  Log file:    {r['log_file']}")
    print()


def cmd_output(args) -> None:
    con = db_open()
    r = con.execute("SELECT log_file, status FROM tasks WHERE id=?", (args.id,)).fetchone()
    if not r:
        print(f"Task {args.id} not found.")
        sys.exit(1)
    if not r["log_file"] or not Path(r["log_file"]).exists():
        print(f"No output log for task {args.id} yet.")
        sys.exit(1)
    print(Path(r["log_file"]).read_text())


def cmd_follow(args) -> None:
    con = db_open()
    r = con.execute("SELECT log_file, status FROM tasks WHERE id=?", (args.id,)).fetchone()
    if not r:
        print(f"Task {args.id} not found.")
        sys.exit(1)

    log = r["log_file"]
    status = r["status"]

    # Wait for the log file to appear if task is queued/starting
    if not log or not Path(log).exists():
        if status in ("queued", "running"):
            print(f"Task {args.id} has not started yet. Waiting...", flush=True)
            for _ in range(60):   # wait up to 60 s
                time.sleep(1)
                r = con.execute(
                    "SELECT log_file, status FROM tasks WHERE id=?", (args.id,)
                ).fetchone()
                log = r["log_file"]
                if log and Path(log).exists():
                    break
            else:
                print("Timed out waiting for task to start.")
                sys.exit(1)
        else:
            print(f"No output log for task {args.id}.")
            sys.exit(1)

    # tail -f equivalent
    print(f"Following task {args.id} (Ctrl+C to stop)...\n")
    try:
        subprocess.run(["tail", "-n", "+1", "-f", log])
    except KeyboardInterrupt:
        pass


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
    return False, f"could not reach {host.name}: {reason}"


def _unconfirmed_kill_warning(row, msg: str) -> str:
    check = f"task --ssh {row['server']}"
    pgid = row["remote_pid"]
    return (f"Warning: {msg}. The process may still be running; check with "
            f"`{check}`" + (f" and `kill -- -{pgid}`." if pgid else "."))


def cmd_kill(args) -> None:
    con = db_open()
    r = con.execute("SELECT * FROM tasks WHERE id=?", (args.id,)).fetchone()
    if not r:
        print(f"Task {args.id} not found.")
        sys.exit(1)

    status = r["status"]
    if status in ("done", "failed", "killed", "cancelled", "lost"):
        print(f"Task {args.id} is already {status}.")
        return

    if status == "queued":
        con.execute(
            "UPDATE tasks SET status='cancelled', finished_at=? WHERE id=? AND status='queued'",
            (time.time(), args.id),
        )
        con.commit()
        print(f"Task {args.id} cancelled.")
        return

    if status == "running":
        # Mark killed first so the daemon doesn't treat the signal as a
        # failure and retry it.
        con.execute(
            "UPDATE tasks SET status='killed', finished_at=?, ssh_pid=NULL WHERE id=?",
            (time.time(), args.id),
        )
        con.commit()
        ok, msg = _kill_remote(r)
        print(msg if ok else _col(_unconfirmed_kill_warning(r, msg), YELLOW))
        print(f"Task {args.id} marked as killed.")


def cmd_kill_all(args) -> None:
    con = db_open()
    now = time.time()

    # Cancel all queued tasks
    n_queued = con.execute(
        "UPDATE tasks SET status='cancelled', finished_at=? WHERE status='queued'",
        (now,),
    ).rowcount

    # Kill all running tasks
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
    for r, (ok, msg) in zip(running, results):
        if not ok:
            print(_col(f"Task {r['id']}: " + _unconfirmed_kill_warning(r, msg), YELLOW))

    print(f"Cancelled {n_queued} queued task(s), killed {n_running} running task(s).")

    if daemon_is_running():
        stop_daemon()

    # Wipe all tasks and reset the ID counter so the next run starts from 1
    con.execute("DELETE FROM tasks")
    con.execute("DELETE FROM sqlite_sequence WHERE name='tasks'")
    con.commit()
    print("Task history cleared. Next task will be ID 1.")


def cmd_clear(args) -> None:
    con = db_open()
    finished = "status IN ('done','failed','killed','cancelled','lost')"
    # Get logs and job directories to delete before removing from DB
    to_delete = con.execute(f"SELECT id, server, log_file FROM tasks WHERE {finished}").fetchall()

    n = con.execute(f"DELETE FROM tasks WHERE {finished}").rowcount
    con.commit()

    # Delete the physical files
    deleted_files = 0
    for r in to_delete:
        if r["log_file"]:
            p = Path(r["log_file"])
            if p.exists():
                try:
                    p.unlink()
                    deleted_files += 1
                except OSError:
                    pass

    # Remove job directories on the hosts (best effort).
    cfg = get_cfg()
    by_host: dict[str, list[int]] = {}
    for r in to_delete:
        if r["server"] in cfg.hosts:
            by_host.setdefault(r["server"], []).append(r["id"])
    cleaned = get_transport().fan_out(
        [cfg.hosts[h] for h in by_host],
        lambda h: get_transport().run(h, jobs.cleanup_script(by_host[h.name]), timeout=30)[0] == 0,
    )
    failed = sorted(h for h, ok in cleaned.items() if not ok)

    print(f"Removed {n} finished task(s) and {deleted_files} log file(s)."
          + (_col(f"  Could not clean job directories on: {', '.join(failed)}", DIM)
             if failed else ""))


def cmd_wait(args) -> None:
    con = db_open()
    print(f"Waiting for task {args.id} to finish (Ctrl+C to stop waiting)...")
    try:
        while True:
            r = con.execute("SELECT status FROM tasks WHERE id=?", (args.id,)).fetchone()
            if not r:
                print(f"Task {args.id} not found.")
                sys.exit(1)
            if r["status"] not in ("queued", "running"):
                print(f"Task {args.id} finished: {r['status']}")
                return
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopped waiting (task still running).")


def _run_on_server(host, command: str) -> tuple[bool, str]:
    """Run command on one host in a login shell; return (success, combined output)."""
    inner = "source ~/.bashrc 2>/dev/null || true; " + command
    rc, out, err = get_transport().run(host, f"exec bash -l -c {shlex.quote(inner)} < /dev/null\n",
                                 timeout=300)
    if rc == 124:
        return False, "timed out after 300s"
    output = (out + err).strip()
    hint = auth_hint(err) if rc == 255 else None
    return rc == 0, f"{output}\n({hint})" if hint else output


def cmd_run_all(args) -> None:
    command = args.run_all
    hosts = get_cfg().workers()
    n = len(hosts)
    print(f"{BOLD}Running on {n} servers:{RESET} {command}  "
          f"{DIM}network: {get_transport().describe_mode()}{RESET}\n")

    results: dict[str, tuple[bool, str]] = {}
    lock = threading.Lock()

    def one(host):
        ok, output = _run_on_server(host, command)
        with lock:
            results[host.name] = (ok, output)
            status = _col("OK  ", GREEN) if ok else _col("FAIL", RED)
            # Show the last non-empty line as a quick summary
            last = next((l for l in reversed(output.splitlines()) if l.strip()), "")
            print(f"  [{len(results):>2}/{n}] {host.name:<16} {status}  {_truncate(last, 60)}",
                  flush=True)

    get_transport().fan_out(hosts, one)

    ok_count   = sum(1 for ok, _ in results.values() if ok)
    fail_count = n - ok_count

    # Print full output for failed servers
    failures = [(s, out) for s, (ok, out) in results.items() if not ok]
    if failures:
        print(f"\n{_col(f'{fail_count} server(s) failed — full output:', RED)}")
        for server, out in failures:
            print(f"\n{BOLD}── {server} ──{RESET}")
            print(out or "(no output)")

    print(
        f"\n{_col(str(ok_count), GREEN)}/{n} servers succeeded"
        + (f"  {_col(str(fail_count) + ' failed', RED)}" if fail_count else "")
        + "\n"
    )


def cmd_hosts(args) -> None:
    hosts = [h for h in get_cfg().hosts.values() if h.role == "worker"]
    mode = get_transport().mode()
    print(f"{BOLD}Querying {len(hosts)} host(s)...{RESET}", flush=True)
    results = query_hosts(get_transport(), [h for h in hosts if h.enabled])

    hdr = (f"{'HOST':<14} {'KIND':<5} {'GROUP':<8} {'VIA':<16} {'STATUS':<11} "
           f"{'GPUS':>4} {'CPU':>3}  PATH MAP")
    print(f"\n{BOLD}{hdr}{RESET}")
    print("─" * max(len(hdr), 80))
    for h in hosts:
        chain = get_cfg().chain(h)
        via = "local" if h.is_local else (
            "→".join(x.alias for x in chain) if (chain and mode == "jump") else "direct")
        r = results.get(h.name)
        if not h.enabled:
            status = _col(f"{'disabled':<11}", DIM)
        elif r and r.reachable:
            status = _col(f"{'ok':<11}", GREEN)
        else:
            status = _col(f"{'unreachable':<11}", RED)
        gpus = str(r.total_gpus) if r and r.reachable else "-"
        pmap = ", ".join(f"{l} → {rem}" for l, rem in h.path_map) or "-"
        print(f"{h.name:<14} {h.kind:<5} {h.group or '-':<8} {_truncate(via, 16):<16} "
              f"{status} {gpus:>4} {h.cpu_slots or '-':>3}  {pmap}")
        if r and not r.reachable and r.error:
            print(f"{'':14} {_col(_truncate(r.error, 90), DIM)}")
    print("─" * max(len(hdr), 80))
    print(f"network: {get_transport().describe_mode()}   config: {get_cfg().source_label}\n")


def _lookup_host(name: str):
    try:
        return get_cfg().get(name)
    except ConfigError as e:
        sys.exit(f"task: {e}")


def cmd_ssh(args) -> None:
    host = _lookup_host(args.ssh)
    argv = get_transport().interactive_argv(host)
    os.execvp(argv[0], argv)


def cmd_connect(args) -> None:
    mode = get_transport().mode(refresh=True)
    if mode != "jump":
        print(f"Network mode is {get_transport().describe_mode()}; no jump hosts to connect.")
        return
    failures = get_transport().connect(get_cfg().workers())
    if failures:
        sys.exit(1)
    print("Jump connections are open; the daemon will reuse them.")


def cmd_init_config(args) -> None:
    path = config_path()
    if path.exists():
        print(f"{path} already exists; not overwriting.")
        sys.exit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config_template())
    print(f"Wrote {path}. Edit it, then check with: task --hosts")


def cmd_gpu_status(args) -> None:
    from gpu_spooler.gpu_status import main as _gpu_status_main
    _gpu_status_main([])

# ── Argument parsing ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="task",
        description="GPU task spooler for SSH-reachable GPU servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        add_help=True,
    )

    # ── Submission ────────────────────────────────────────────────────────────
    p.add_argument(
        "-G", "--gpus", metavar="N", type=int,
        help="Number of GPUs required (submit mode); 0 for a CPU-only job",
    )
    p.add_argument(
        "-H", "--host", metavar="SPEC",
        help="Only run on these hosts, groups or tags (comma separated)",
    )
    p.add_argument(
        "-m", "--min-mem", metavar="GB", type=int, default=0,
        help="Minimum VRAM per GPU in GB (e.g. 40 for A40/A6000)",
    )
    p.add_argument(
        "-n", "--name", metavar="LABEL",
        help="Optional human-readable label for the task",
    )
    p.add_argument(
        "-d", "--dir", metavar="PATH",
        help="Working directory (default: current directory; translated by path_map)",
    )

    # ── Inspection ────────────────────────────────────────────────────────────
    p.add_argument("-l", "--list",   action="store_true", help="List tasks (most recent 200)")
    p.add_argument("-L", "--all",    action="store_true", help="List all tasks (no limit)")
    p.add_argument("-i", "--info",   metavar="ID", type=int, help="Detailed info for a task")
    p.add_argument("-o", "--output", metavar="ID", type=int, help="Print saved output of a task")
    p.add_argument("-f", "--follow", metavar="ID", type=int, help="Stream live output of a task")

    # ── Management ───────────────────────────────────────────────────────────
    p.add_argument("-k",  "--kill",     metavar="ID", type=int, help="Kill or cancel a task")
    p.add_argument("-K", "--kill-all", action="store_true",    help="Kill all running and queued tasks and stop the daemon")
    p.add_argument("-c", "--clear", action="store_true", help="Remove finished tasks from the list")
    p.add_argument("-w", "--wait",  metavar="ID", type=int, help="Block until a task finishes")

    # ── GPU / daemon ──────────────────────────────────────────────────────────
    p.add_argument("-R", "--run-all", metavar="CMD",
                   help="Run a shell command on every GPU server in parallel and report results")
    p.add_argument("--gpu-status", action="store_true", help="Show GPU availability across servers")
    p.add_argument("--daemon",     action="store_true", help="Start daemon (if not already running)")
    p.add_argument("--stop",       action="store_true", help="Stop the background daemon")
    p.add_argument("--hosts",      action="store_true",
                   help="Show configured hosts, reachability and network mode")
    p.add_argument("--connect",    action="store_true",
                   help="Open jump-host connections interactively (password/OTP once)")
    p.add_argument("--ssh",        metavar="HOST",
                   help="Open an interactive shell on a host via the generated ssh config")
    p.add_argument("--init-config", action="store_true",
                   help="Write a commented hosts.toml template (never overwrites)")

    # Internal: daemon loop (not for user use)
    p.add_argument("--_daemon-loop", action="store_true", help=argparse.SUPPRESS)

    # Positional: the command to run (used with -G)
    p.add_argument("command", nargs=argparse.REMAINDER, help="Command to queue (used with -G)")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Internal: run the actual daemon loop
    if args._daemon_loop:
        Daemon().run()
        return

    # Route to the appropriate command
    if args.gpus is not None:
        if not args.command:
            parser.error("-G requires a command, e.g.: task -G 1 python3 train.py")
        cmd_submit(args)
    elif args.list or args.all:
        cmd_list(args)
    elif args.info is not None:
        args.id = args.info;   cmd_info(args)
    elif args.output is not None:
        args.id = args.output; cmd_output(args)
    elif args.follow is not None:
        args.id = args.follow; cmd_follow(args)
    elif args.kill is not None:
        args.id = args.kill;   cmd_kill(args)
    elif args.kill_all:
        cmd_kill_all(args)
    elif args.clear:
        cmd_clear(args)
    elif args.wait is not None:
        args.id = args.wait;   cmd_wait(args)
    elif args.run_all:
        cmd_run_all(args)
    elif args.gpu_status:
        cmd_gpu_status(args)
    elif args.daemon:
        start_daemon()
    elif args.stop:
        stop_daemon()
    elif args.init_config:
        cmd_init_config(args)
    elif args.hosts:
        cmd_hosts(args)
    elif args.ssh:
        cmd_ssh(args)
    elif args.connect:
        cmd_connect(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
