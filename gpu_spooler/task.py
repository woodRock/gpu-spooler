#!/usr/bin/env python3
"""
task - GPU task spooler for VUW ECS GPU servers

The background daemon auto-starts on first job submission and dispatches
queued tasks to GPU servers as GPUs become free.

Submission:
  task -G 1 python3 train.py             queue a job needing 1 GPU
  task -G 2 -n "experiment A" ./run.sh   queue with a label

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

GPU / daemon:
  task --gpu-status                      show GPU availability across all servers
  task --stop                            stop the background daemon
  task --daemon                          (re)start the daemon explicitly
"""

import argparse
import concurrent.futures
import os
import shlex
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────

STATE_DIR  = Path.home() / ".task-spooler"
DB_PATH    = STATE_DIR / "tasks.db"
PID_FILE   = STATE_DIR / "daemon.pid"
DAEMON_LOG = STATE_DIR / "daemon.log"
LOGS_DIR   = STATE_DIR / "logs"

POLL_INTERVAL = 30   # seconds between GPU polls in daemon
SSH_TIMEOUT   = 15   # seconds for GPU-check SSH calls

# ── Hosts and GPU querying ────────────────────────────────────────────────────

from gpu_spooler.config import ConfigError, config_path, config_template, load_config
from gpu_spooler.gpu import ServerResult, query_hosts
from gpu_spooler.transport import Transport, auth_hint

CFG = load_config()
TRANSPORT = Transport(CFG)
SERVERS = [h.name for h in CFG.workers()]
SERVER_INFO = {
    h.name: {"gpu_type": h.gpu_type, "num_gpus": h.num_gpus, "gpu_mem_gb": h.gpu_mem_gb}
    for h in CFG.workers()
}


def query_all_servers() -> dict[str, dict[int, bool]]:
    """Query all servers in parallel. Returns {server: {gpu_idx: is_free}}."""
    results = query_hosts(TRANSPORT, CFG.workers())
    return {name: {g.index: g.is_free for g in r.gpus}
            for name, r in results.items() if r.reachable}


def assign_tasks(task_rows, server_gpus: dict[str, dict[int, bool]]) -> list[tuple]:
    """
    Greedily assign tasks (in queue order) to servers with free GPUs.
    Mutates server_gpus to mark assigned GPUs as reserved.
    Returns list of (task_id, server, [gpu_indices]).
    """
    # Sort servers: prefer those with fewer total GPUs (bin-pack fuller servers first)
    servers_by_size = sorted(
        server_gpus.keys(),
        key=lambda s: SERVER_INFO.get(s, {}).get("num_gpus", 99),
    )
    assignments = []
    for row in task_rows:
        n = row["num_gpus"]
        try:
            min_mem = row["min_mem"]
        except (IndexError, KeyError, sqlite3.OperationalError):
            min_mem = 0
        min_mem = min_mem or 0
        
        for server in servers_by_size:
            info = SERVER_INFO.get(server, {})
            server_mem = info.get("gpu_mem_gb", 0)
            if server_mem < min_mem:
                continue

            gpus_dict = server_gpus[server]
            free = [i for i, ok in sorted(gpus_dict.items()) if ok]
            if len(free) >= n:
                chosen = free[:n]
                for idx in chosen:
                    server_gpus[server][idx] = False   # reserve
                assignments.append((row["id"], server, chosen))
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

MAX_RETRIES = 3


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
    
    if "retry_count" not in cols:
        con.execute("ALTER TABLE tasks ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0")
        con.commit()
    if "min_mem" not in cols:
        con.execute("ALTER TABLE tasks ADD COLUMN min_mem INTEGER NOT NULL DEFAULT 0")
        con.commit()
    
    return con

# ── Daemon ────────────────────────────────────────────────────────────────────

class Daemon:
    """
    Background process that dispatches queued tasks to GPU servers
    and monitors running SSH subprocesses for completion.
    """

    def __init__(self):
        self.con = db_open()
        # {task_id: (proc, log_file_handle)}
        self._running: dict[int, tuple[subprocess.Popen, object]] = {}

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    def run(self) -> None:
        self._log("Daemon started.")
        # Mark any tasks left in 'running' state (from a previous daemon) as lost
        n = self.con.execute(
            "UPDATE tasks SET status='lost', finished_at=? WHERE status='running'",
            (time.time(),),
        ).rowcount
        if n:
            self._log(f"Marked {n} orphaned task(s) as 'lost'.")
        self.con.commit()

        while True:
            try:
                self._reap_finished()
                self._dispatch()
            except Exception as e:
                self._log(f"Error in daemon loop: {e}")
            time.sleep(POLL_INTERVAL)

    def _reap_finished(self) -> None:
        """Check running SSH subprocesses for completion."""
        done = []
        for tid, (proc, log_fh) in self._running.items():
            rc = proc.poll()
            if rc is not None:
                log_fh.close()
                if rc == 0:
                    self.con.execute(
                        "UPDATE tasks SET status='done', exit_code=?, finished_at=?, ssh_pid=NULL "
                        "WHERE id=?",
                        (rc, time.time(), tid),
                    )
                    self.con.commit()
                    self._log(f"Task {tid} finished — exit {rc} (done).")
                else:
                    row = self.con.execute(
                        "SELECT retry_count FROM tasks WHERE id=?", (tid,)
                    ).fetchone()
                    retry_count = row["retry_count"] if row else MAX_RETRIES
                    if retry_count < MAX_RETRIES:
                        self.con.execute(
                            "UPDATE tasks SET status='queued', exit_code=?, ssh_pid=NULL, "
                            "started_at=NULL, finished_at=NULL, server=NULL, gpu_indices=NULL, "
                            "retry_count=retry_count+1 WHERE id=?",
                            (rc, tid),
                        )
                        self.con.commit()
                        self._log(
                            f"Task {tid} failed (exit {rc}), "
                            f"re-queuing (attempt {retry_count + 1}/{MAX_RETRIES})."
                        )
                    else:
                        self.con.execute(
                            "UPDATE tasks SET status='failed', exit_code=?, finished_at=?, "
                            "ssh_pid=NULL WHERE id=?",
                            (rc, time.time(), tid),
                        )
                        self.con.commit()
                        self._log(
                            f"Task {tid} failed (exit {rc}), "
                            f"no more retries ({MAX_RETRIES}/{MAX_RETRIES})."
                        )
                done.append(tid)
        for tid in done:
            del self._running[tid]

    def _dispatch(self) -> None:
        """Query servers and start queued tasks on free GPUs."""
        rows = self.con.execute(
            "SELECT * FROM tasks WHERE status='queued' ORDER BY id"
        ).fetchall()
        if not rows:
            return

        self._log(f"Polling GPU servers for {len(rows)} queued task(s)...")
        server_gpus = query_all_servers()
        
        # Filter out unreachable servers (those that returned None)
        reachable = {s: g for s, g in server_gpus.items() if g is not None}
        if not reachable:
            self._log("No reachable servers (check SSH keys/agent).")
            return

        # Mark GPUs claimed by our own running tasks as busy
        running = self.con.execute(
            "SELECT server, gpu_indices FROM tasks WHERE status='running'"
        ).fetchall()
        for r in running:
            if r["server"] in reachable and r["gpu_indices"]:
                for idx_str in r["gpu_indices"].split(","):
                    try:
                        reachable[r["server"]][int(idx_str)] = False
                    except (ValueError, KeyError):
                        pass

        # Summary of availability
        avail_msg = ", ".join(f"{s}:{sum(g.values())}" for s, g in sorted(reachable.items()))
        self._log(f"Free GPUs: {avail_msg}")

        assignments = assign_tasks(rows, reachable)
        if not assignments:
            self._log("No free GPU slots available; tasks remain queued.")
            return

        for task_id, server, gpu_indices in assignments:
            self._launch(task_id, server, gpu_indices)

    def _launch(self, task_id: int, server: str, gpu_indices: list[int]) -> None:
        row = self.con.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            return

        gpu_str  = ",".join(str(i) for i in gpu_indices)
        log_path = LOGS_DIR / f"{task_id}.log"

        # Reserve in DB immediately
        self.con.execute(
            "UPDATE tasks SET status='running', server=?, gpu_indices=?, "
            "started_at=?, log_file=? WHERE id=?",
            (server, gpu_str, time.time(), str(log_path), task_id),
        )
        self.con.commit()

        # Write a header to the log
        with open(log_path, "w") as f:
            f.write(
                f"# task {task_id}: {row['command']}\n"
                f"# server: {server}  CUDA_VISIBLE_DEVICES: {gpu_str}\n"
                f"# started: {datetime.now().isoformat()}\n"
                f"# working_dir: {row['working_dir']}\n"
                f"{'─' * 60}\n"
            )

        wdir = row['working_dir']
        # `ls` the path first so autofs/NFS automounts have a chance to mount
        # the filesystem before the cd. Without this, non-interactive SSH
        # sessions see the directory as missing on automount-based systems.
        # Added a short sleep for multi-GPU tasks to reduce chance of torchrun
        # port collisions if multiple tasks are launched simultaneously.
        delay = "sleep 2; " if len(gpu_indices) > 1 else ""
        inner = (
            f"ls {wdir!r} > /dev/null 2>&1; "
            f"cd {wdir!r} && "
            f"{delay}"
            f"export CUDA_DEVICE_ORDER=PCI_BUS_ID && "
            f"export CUDA_VISIBLE_DEVICES={gpu_str} && "
            f"{row['command']}"
        )
        # Use a login shell AND explicitly source ~/.bashrc.
        # bash -l sources /etc/profile and ~/.bash_profile, but on most Linux
        # systems ~/.local/bin (where pip --user installs) is only added to
        # PATH inside ~/.bashrc, which is only sourced for interactive shells.
        # Sourcing it explicitly here ensures user-installed commands are found.
        wrapped = f"source ~/.bashrc 2>/dev/null || true; {inner}"
        ssh_cmd = f"bash -l -c {shlex.quote(wrapped)}"
        try:
            log_fh = open(log_path, "a")
            proc = subprocess.Popen(
                ["ssh",
                 "-o", "BatchMode=yes",
                 "-o", "StrictHostKeyChecking=no",
                 "-o", "ServerAliveInterval=60",
                 "-o", "ServerAliveCountMax=3",
                 server, ssh_cmd],
                stdout=log_fh,
                stderr=log_fh,
            )
            self._running[task_id] = (proc, log_fh)
            self.con.execute(
                "UPDATE tasks SET ssh_pid=? WHERE id=?", (proc.pid, task_id)
            )
            self.con.commit()
            self._log(
                f"Task {task_id} dispatched → {server} "
                f"[CUDA_VISIBLE_DEVICES={gpu_str}]  (SSH PID {proc.pid})"
            )
        except Exception as e:
            log_fh.close()
            row = self.con.execute(
                "SELECT retry_count FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            retry_count = row["retry_count"] if row else MAX_RETRIES
            if retry_count < MAX_RETRIES:
                self.con.execute(
                    "UPDATE tasks SET status='queued', ssh_pid=NULL, "
                    "started_at=NULL, finished_at=NULL, server=NULL, gpu_indices=NULL, "
                    "retry_count=retry_count+1 WHERE id=?",
                    (task_id,),
                )
                self.con.commit()
                self._log(
                    f"Task {task_id} failed to launch: {e} — "
                    f"re-queuing (attempt {retry_count + 1}/{MAX_RETRIES})."
                )
            else:
                self.con.execute(
                    "UPDATE tasks SET status='failed', finished_at=? WHERE id=?",
                    (time.time(), task_id),
                )
                self.con.commit()
                self._log(
                    f"Task {task_id} failed to launch: {e} — "
                    f"no more retries ({MAX_RETRIES}/{MAX_RETRIES})."
                )

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
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--_daemon-loop"],
        stdout=open(DAEMON_LOG, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,   # detach from terminal
        close_fds=True,
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
        print(f"Daemon (PID {pid}) stopped.")
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

def cmd_submit(args) -> None:
    command = shlex.join(args.command)
    working_dir = os.path.abspath(args.dir) if args.dir else os.getcwd()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    con = db_open()
    cur = con.execute(
        "INSERT INTO tasks (command, num_gpus, min_mem, submitted_at, working_dir, name) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (command, args.gpus, args.min_mem, time.time(), working_dir, args.name),
    )
    con.commit()
    task_id = cur.lastrowid
    label = f'  "{args.name}"' if args.name else ""
    print(f"Queued task {task_id}{label}: {command}")
    print(f"  Requires: {args.gpus} GPU(s) (min {args.min_mem or 0}GB VRAM)   Working dir: {working_dir}")

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
        f"{'ID':>4}  {'STATUS':10}  {'G':>1}  {'SERVER':<14}  "
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
        server  = r["server"] or "-"
        gpus    = r["gpu_indices"] or "-"

        print(
            f"{r['id']:>4}  {_col(f'{status:10}', color)}  {r['num_gpus']:>1}  "
            f"{server:<14}  {gpus:>4}  {_fmt_ts(r['submitted_at']):11}  "
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
        f"daemon: {daemon_str}\n"
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
    print(f"  GPUs req:    {r['num_gpus']}")
    print(f"  Working dir: {r['working_dir']}")
    print(f"  Submitted:   {_fmt_ts(r['submitted_at'])}")
    if r["server"]:
        print(f"  Server:      {r['server']}")
        info = SERVER_INFO.get(r["server"], {})
        if info:
            print(f"               ({info.get('gpu_type','')}  {info.get('gpu_mem_gb','')} GB/GPU)")
    if r["gpu_indices"]:
        print(f"  GPU indices: {r['gpu_indices']}")
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
        proc = subprocess.run(["tail", "-n", "+1", "-f", log])
    except KeyboardInterrupt:
        pass


def cmd_kill(args) -> None:
    con = db_open()
    r = con.execute("SELECT * FROM tasks WHERE id=?", (args.id,)).fetchone()
    if not r:
        print(f"Task {args.id} not found.")
        sys.exit(1)

    status = r["status"]
    if status in ("done", "failed", "killed", "cancelled"):
        print(f"Task {args.id} is already {status}.")
        return

    if status == "queued":
        con.execute(
            "UPDATE tasks SET status='cancelled', finished_at=? WHERE id=?",
            (time.time(), args.id),
        )
        con.commit()
        print(f"Task {args.id} cancelled.")
        return

    if status == "running":
        ssh_pid = r["ssh_pid"]
        if ssh_pid:
            try:
                os.kill(ssh_pid, signal.SIGTERM)
                print(f"Sent SIGTERM to SSH client (PID {ssh_pid}).")
            except ProcessLookupError:
                print("SSH process already gone.")
        con.execute(
            "UPDATE tasks SET status='killed', finished_at=?, ssh_pid=NULL WHERE id=?",
            (time.time(), args.id),
        )
        con.commit()
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
    running = con.execute(
        "SELECT id, ssh_pid FROM tasks WHERE status='running'"
    ).fetchall()
    for r in running:
        if r["ssh_pid"]:
            try:
                os.kill(r["ssh_pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
    n_running = con.execute(
        "UPDATE tasks SET status='killed', finished_at=?, ssh_pid=NULL WHERE status='running'",
        (now,),
    ).rowcount
    con.commit()

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
    # Get logs to delete before removing from DB
    to_delete = con.execute(
        "SELECT log_file FROM tasks WHERE status IN ('done','failed','killed','cancelled','lost')"
    ).fetchall()
    
    n = con.execute(
        "DELETE FROM tasks WHERE status IN ('done','failed','killed','cancelled','lost')"
    ).rowcount
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
                    
    print(f"Removed {n} finished task(s) and {deleted_files} log file(s).")


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
    rc, out, err = TRANSPORT.run(host, f"exec bash -l -c {shlex.quote(inner)} < /dev/null\n",
                                 timeout=300)
    if rc == 124:
        return False, "timed out after 300s"
    output = (out + err).strip()
    hint = auth_hint(err) if rc == 255 else None
    return rc == 0, f"{output}\n({hint})" if hint else output


def cmd_run_all(args) -> None:
    command = args.run_all
    hosts = CFG.workers()
    n = len(hosts)
    print(f"{BOLD}Running on {n} servers:{RESET} {command}  "
          f"{DIM}network: {TRANSPORT.describe_mode()}{RESET}\n")

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

    TRANSPORT.fan_out(hosts, one)

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
    hosts = [h for h in CFG.hosts.values() if h.role == "worker"]
    mode = TRANSPORT.mode()
    print(f"{BOLD}Querying {len(hosts)} host(s)...{RESET}", flush=True)
    results = query_hosts(TRANSPORT, [h for h in hosts if h.enabled])

    hdr = (f"{'HOST':<14} {'KIND':<5} {'GROUP':<8} {'VIA':<16} {'STATUS':<11} "
           f"{'GPUS':>4} {'CPU':>3}  PATH MAP")
    print(f"\n{BOLD}{hdr}{RESET}")
    print("─" * max(len(hdr), 80))
    for h in hosts:
        chain = CFG.chain(h)
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
    print(f"network: {TRANSPORT.describe_mode()}   config: {CFG.source_label}\n")


def _lookup_host(name: str):
    try:
        return CFG.get(name)
    except ConfigError as e:
        sys.exit(f"task: {e}")


def cmd_ssh(args) -> None:
    host = _lookup_host(args.ssh)
    argv = TRANSPORT.interactive_argv(host)
    os.execvp(argv[0], argv)


def cmd_connect(args) -> None:
    mode = TRANSPORT.mode(refresh=True)
    if mode != "jump":
        print(f"Network mode is {TRANSPORT.describe_mode()}; no jump hosts to connect.")
        return
    failures = TRANSPORT.connect(CFG.workers())
    if failures:
        sys.exit(1)
    print("Jump connections are open; the daemon will reuse them.")


def cmd_gpu_status(args) -> None:
    from gpu_spooler.gpu_status import main as _gpu_status_main
    _gpu_status_main([])

# ── Argument parsing ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="task",
        description="GPU task spooler for VUW ECS GPU servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        add_help=True,
    )

    # ── Submission ────────────────────────────────────────────────────────────
    p.add_argument(
        "-G", "--gpus", metavar="N", type=int,
        help="Number of GPUs required (submit mode)",
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
        help="Working directory on the remote server (default: current directory)",
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
    if args._daemon_loop:
        Daemon().run()
    elif args.gpus is not None:
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
