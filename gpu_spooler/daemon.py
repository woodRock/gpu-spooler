"""
The gpu-spooler daemon: dispatches queued tasks to hosts and polls detached
jobs for completion, syncing their logs. Started by `core.start_daemon()`
(python -m gpu_spooler.daemon).
"""

import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from gpu_spooler import core, jobs, provenance
from gpu_spooler.config import Config, ConfigError, Host, config_path, load_config
from gpu_spooler.gpu import ServerResult, gpu_type_matches, query_hosts, write_snapshot
from gpu_spooler.transport import auth_hint

POLL_INTERVAL   = 30   # seconds between GPU polls / dispatch
STATUS_INTERVAL = float(os.environ.get("GPU_SPOOLER_STATUS_INTERVAL", 5))
                       # seconds between status / log-sync polls of running jobs
LAUNCH_TIMEOUT  = 60
STATUS_TIMEOUT  = 60
SNAPSHOT_WANTED = 120  # keep refreshing the GPU snapshot this long after a request

PACKAGE_DIR = Path(__file__).resolve().parent


def code_signature() -> tuple:
    """Changes whenever a source file of the package is edited or updated."""
    return tuple(sorted((p.name, p.stat().st_mtime) for p in PACKAGE_DIR.glob("*.py")))

# ── Assignment ────────────────────────────────────────────────────────────────


@dataclass
class HostState:
    """What a host can take right now, for one dispatch round."""
    host: Host
    free: dict[int, bool]          # gpu index → free
    cpu_free: int                  # unused CPU slots
    detected_mem_gb: float = 0.0   # smallest detected GPU memory (decimal GB)
    names: dict[int, str] = field(default_factory=dict)   # gpu index → nvidia-smi name

    @property
    def mem_gb(self) -> float:
        """Detected memory when the host reported GPUs; the configured value otherwise."""
        if self.detected_mem_gb or self.host.gpu_mem_gb is None:
            return self.detected_mem_gb
        return self.host.gpu_mem_gb


def _field(row, key: str):
    """row[key], or None for rows (or test dicts) without that column."""
    return row[key] if key in row.keys() else None


def gpu_budget(cfg: Config, running_rows) -> Optional[int]:
    """GPUs our tasks may still take under [limits] max_gpus (None: no limit)."""
    if cfg.max_gpus is None:
        return None
    return max(0, cfg.max_gpus - sum(r["num_gpus"] or 0 for r in running_rows))


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
            names={g.index: g.name for g in r.gpus},
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


def assign_tasks(task_rows, states: dict[str, HostState], cfg: Config,
                 budget: Optional[int] = None) -> list[tuple]:
    """
    Greedily assign tasks (in queue order) to hosts. GPU tasks bin-pack onto
    hosts with the fewest GPUs first, using only GPUs whose nvidia-smi name
    matches the task's --gpu-type; -G 0 tasks use free CPU slots. budget caps the
    GPUs handed out in this round ([limits] max_gpus minus those in use).
    Mutates states to reserve what is assigned.
    Returns list of (task_id, host_name, [gpu_indices]).
    """
    order = sorted(states.values(), key=lambda s: len(s.free))
    assignments = []
    for row in task_rows:
        n = row["num_gpus"]
        if budget is not None and n > budget:
            continue
        min_mem = row["min_mem"] or 0
        gpu_type = _field(row, "gpu_type")
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
            free = [i for i, ok in sorted(st.free.items())
                    if ok and gpu_type_matches(gpu_type, st.names.get(i, ""))]
            if len(free) >= n:
                chosen = free[:n]
                for idx in chosen:
                    st.free[idx] = False   # reserve
                assignments.append((row["id"], st.host.name, chosen))
                if budget is not None:
                    budget -= n
                break
    return assignments

class Daemon:
    """
    Background process that dispatches queued tasks to hosts and polls the
    detached jobs for completion, syncing their logs. All job state lives in
    the database and the job directories, so a restarted daemon re-adopts
    jobs that are still running.
    """

    def __init__(self):
        self.con = core.db_open()
        self.cfg = core.get_cfg()
        self.transport = core.get_transport()
        self._last_snapshot = 0.0
        self._code = code_signature()
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
        self._maybe_restart_on_code_change()
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

        if self._snapshot_due():
            self._write_snapshot(query_hosts(self.transport, self.cfg.workers()))

    # ── GPU snapshot for viewers (tui) ───────────────────────────────────────

    def _snapshot_due(self) -> bool:
        """A viewer asked recently and the snapshot is older than its request or stale."""
        try:
            requested = core.SNAPSHOT_REQUEST.stat().st_mtime
        except OSError:
            return False
        now = time.time()
        if now - requested > SNAPSHOT_WANTED:
            return False
        return requested > self._last_snapshot or now - self._last_snapshot >= POLL_INTERVAL

    def _write_snapshot(self, results: dict[str, ServerResult]) -> None:
        self._last_snapshot = time.time()
        try:
            write_snapshot(core.SNAPSHOT_PATH, results, self.transport.mode(),
                           keep=[h.name for h in self.cfg.workers()])
        except OSError as e:
            self._log(f"Cannot write GPU snapshot: {e}")

    def _maybe_restart_on_code_change(self) -> None:
        """
        Re-exec with the current code after an update (editable installs pick
        up edits immediately). Same PID, so the PID file stays valid; running
        jobs are re-adopted from the database like after any restart.
        """
        try:
            changed = code_signature() != self._code
        except OSError:
            return
        if changed:
            self._log("Code changed on disk; restarting the daemon to pick it up.")
            sys.stdout.flush()
            os.execv(sys.executable, [sys.executable, "-m", "gpu_spooler.daemon"])

    def _maybe_reload_config(self) -> None:
        path = config_path()
        try:
            mtime = path.stat().st_mtime if path.exists() else 0.0
        except OSError:
            return
        current = self.cfg.mtime if self.cfg.source else 0.0
        if mtime == current:
            return
        try:
            cfg = load_config()
        except ConfigError as e:
            self._log(f"Config reload failed, keeping previous config: {e}")
            self.cfg.mtime = mtime          # don't retry until it changes again
            return
        core.set_config(cfg)
        self.cfg, self.transport = core.get_cfg(), core.get_transport()
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

    def _provenance(self, event: str, task_id: int, row=None) -> None:
        row = row or self.con.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is not None and provenance.notify(event, row, self.cfg.taura, log=self._log):
            self._log(f"Task {task_id}: recorded {event} as taura run {provenance.run_id(row)}.")

    def _finish(self, row, rc: Optional[int]) -> None:
        tid, key = row["id"], row["job_key"]
        # Record this attempt as it ended, before a retry re-queues it (and bumps its attempt number).
        ended = {**dict(row), "exit_code": rc, "finished_at": time.time()}
        self._provenance("finish", tid, ended)
        if rc == 0:
            if self._update_running(tid, key, "status='done', exit_code=0, finished_at=?",
                                    (time.time(),)):
                self._log(f"Task {tid} finished — exit 0 (done).")
            return
        if (row["retry_count"] or 0) < core.MAX_RETRIES:
            if self._update_running(tid, key, f"{core.REQUEUE}, exit_code=?", (rc,)):
                self._log(f"Task {tid} failed (exit {rc}), re-queuing "
                          f"(attempt {row['retry_count'] + 1}/{core.MAX_RETRIES}).")
        elif self._update_running(tid, key, "status='failed', exit_code=?, finished_at=?",
                                  (rc, time.time())):
            self._log(f"Task {tid} failed (exit {rc}), "
                      f"no more retries ({core.MAX_RETRIES}/{core.MAX_RETRIES}).")

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
        self._write_snapshot(results)

        running = self.con.execute(
            "SELECT server, gpu_indices, num_gpus FROM tasks WHERE status='running'"
        ).fetchall()
        states = host_states(self.cfg, results, running)
        if not states:
            hints = {auth_hint(r.error) for r in results.values()} - {None}
            self._log("No reachable hosts (check SSH keys/agent"
                      + ("; run `task connect`" if hints else "") + ").")
            return

        avail = ", ".join(
            f"{n}:{sum(s.free.values())}" + (f"+{s.cpu_free}cpu" if s.cpu_free > 0 else "")
            for n, s in sorted(states.items()))
        budget = gpu_budget(self.cfg, running)
        cap = "" if budget is None else (
            f"  (max_gpus {self.cfg.max_gpus}: {self.cfg.max_gpus - budget} in use)")
        self._log(f"Free: {avail}{cap}")

        assignments = assign_tasks(rows, states, self.cfg, budget)
        if not assignments:
            self._log("No free slots available"
                      + (" within max_gpus" if budget == 0 else "")
                      + "; tasks remain queued.")
            return

        for task_id, host_name, gpu_indices in assignments:
            self._launch(task_id, self.cfg.hosts[host_name], gpu_indices)

    def _launch(self, task_id: int, host: Host, gpu_indices: list[int]) -> None:
        row = self.con.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            return

        gpu_str  = ",".join(str(i) for i in gpu_indices)
        log_path = core.LOGS_DIR / f"{task_id}.log"
        key      = jobs.new_job_key(task_id)
        wdir     = row["working_dir"]
        rdir     = self.cfg.remote_dir(host, wdir, row["repo_root"])

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
        probe = provenance.probe_command(row["command"]) if provenance.enabled(row, self.cfg.taura) else None
        script = jobs.launch_script(key, rdir, gpu_str, row["command"],
                                    delay=len(gpu_indices) > 1, probe=probe)
        rc, out, err = self.transport.run(host, script, timeout=LAUNCH_TIMEOUT)
        pid = jobs.parse_launch(out)
        if pid:
            git = jobs.parse_git(out)
            self.con.execute("UPDATE tasks SET remote_pid=?, git_commit=?, git_dirty=? WHERE id=? AND job_key=?",
                             (pid, git[0] if git else None, git[1] if git else None, task_id, key))
            self.con.commit()
            self._log(f"Task {task_id} dispatched → {host.name} "
                      f"[CUDA_VISIBLE_DEVICES={gpu_str}]  (job {key}, PGID {pid})")
            self._provenance("start", task_id)
            return
        if rc == 124:
            # The job may or may not have started; the status poll will tell.
            self._log(f"Task {task_id}: launch on {host.name} timed out; "
                      "will check its job directory.")
            return

        reason = (err.strip().splitlines() or [f"exit {rc}"])[-1]
        hint = auth_hint(err)
        with open(log_path, "a") as f:
            f.write(f"gpu-spooler: launch on {host.name} failed: {err.strip() or out.strip()}\n")
        retry_count = row["retry_count"] or 0
        if retry_count < core.MAX_RETRIES:
            self._update_running(task_id, key, core.REQUEUE)
            self._log(f"Task {task_id} failed to launch on {host.name}: {reason}"
                      + (f" ({hint})" if hint else "")
                      + f" — re-queuing (attempt {retry_count + 1}/{core.MAX_RETRIES}).")
        else:
            self._update_running(task_id, key, "status='failed', finished_at=?", (time.time(),))
            self._log(f"Task {task_id} failed to launch on {host.name}: {reason} — "
                      f"no more retries ({core.MAX_RETRIES}/{core.MAX_RETRIES}).")


def main() -> None:
    try:
        Daemon().run()
    except ConfigError as e:
        raise SystemExit(f"task daemon: config error: {e}")


if __name__ == "__main__":
    main()
