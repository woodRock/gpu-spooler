"""
Switching daemons mid-flight, both ways, on the fake cluster: what install.sh does when it
replaces the Python version with the Rust one (and a rollback does in reverse). The daemon that
takes over must adopt the running jobs, finish them and copy the rest of their output, each byte
once. Skipped unless GPU_SPOOLER_BIN points at the Rust `task` binary.
"""

import os
import subprocess
import sys
import time

import pytest

BIN = os.environ.get("GPU_SPOOLER_BIN")
pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not BIN, reason="GPU_SPOOLER_BIN is not set")]

PY = [sys.executable, "-m", "gpu_spooler.task"]


def task(ts, version, *args, check=True):
    argv = PY if version == "python" else [os.path.abspath(BIN)]
    r = subprocess.run([*argv, *args], env=ts.env, capture_output=True, text=True, timeout=120,
                       cwd=str(ts.home))
    if check and r.returncode != 0:
        raise AssertionError(f"{version} task {' '.join(args)}: {r.stdout}{r.stderr}")
    return r


def submit(ts, version, *args) -> int:
    return int(task(ts, version, "submit", "-q", *args).stdout.strip())


def daemon(ts) -> tuple[int, str]:
    """PID and kind ('python' or 'rust') of the running daemon."""
    pid = int((ts.state / "daemon.pid").read_text())
    cmd = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
    if "gpu_spooler.daemon" in cmd:
        return pid, "python"
    if "__daemon" in cmd:
        return pid, "rust"
    raise AssertionError(f"daemon.pid {pid} is not a daemon: {cmd!r}")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop(ts, version):
    pid, _ = daemon(ts)
    task(ts, version, "daemon", "stop")
    deadline = time.time() + 10
    while alive(pid):
        assert time.time() < deadline, f"daemon {pid} did not stop"
        time.sleep(0.1)


def log_body(ts, tid) -> str:
    return (ts.state / "logs" / f"{tid}.log").read_text().split("─" * 60 + "\n", 1)[1]


@pytest.mark.parametrize("first,second", [("python", "rust"), ("rust", "python")])
def test_daemon_handover_adopts_running_jobs(ts, first, second):
    # A job that writes over time, so output arrives before and after the switch.
    script = "for i in 1 2 3 4 5 6 7 8; do echo line$i; sleep 1; done"
    gpu = submit(ts, first, "--", script)
    cpu = submit(ts, first, "-g", "0", "-H", "local", "--", script)
    assert daemon(ts)[1] == first
    for tid in (gpu, cpu):
        ts.wait_status(tid, "running", 30)
    time.sleep(3)                                   # some output synced by the first daemon
    stop(ts, first)
    assert ts.row(gpu)["status"] == "running"
    task(ts, second, "daemon", "start")
    assert daemon(ts)[1] == second
    for tid in (gpu, cpu):
        r = ts.wait_status(tid, ("done", "failed", "lost"), 60)
        assert r["status"] == "done", r
        assert log_body(ts, tid) == "".join(f"line{i}\n" for i in range(1, 9))   # every line, once
    assert "Re-adopting 2 running task(s)." in (ts.state / "daemon.log").read_text()


@pytest.mark.parametrize("submitter,runner", [("rust", "python"), ("python", "rust")])
def test_tasks_submitted_by_one_run_on_the_others_daemon(ts, submitter, runner):
    task(ts, runner, "daemon", "start")
    pid, kind = daemon(ts)
    assert kind == runner
    tid = submit(ts, submitter, "-n", "cross", "-g", "2", "-H", "cuda02", "--", "echo hi; exit 0")
    assert daemon(ts) == (pid, runner)              # the running daemon was kept, not doubled
    r = ts.wait_status(tid, ("done", "failed", "lost"), 60)
    assert r["status"] == "done" and r["name"] == "cross" and len(r["gpu_indices"].split(",")) == 2
    assert log_body(ts, tid) == "hi\n"


@pytest.mark.parametrize("starter,killer", [("python", "rust"), ("rust", "python")])
def test_kill_a_job_the_other_version_started(ts, starter, killer):
    tid = submit(ts, starter, "--", "sleep 3021")
    ts.wait_status(tid, "running", 30)
    deadline = time.time() + 15
    while not ts.row(tid)["remote_pid"]:
        assert time.time() < deadline, "job never started"
        time.sleep(0.2)
    out = task(ts, killer, "kill", str(tid)).stdout
    assert "Sent SIGTERM to process group" in out, out
    deadline = time.time() + 5
    while subprocess.run(["pgrep", "-f", "sleep 3021"], capture_output=True).returncode == 0:
        assert time.time() < deadline, "process group still alive"
        time.sleep(0.2)
    time.sleep(2)
    assert ts.row(tid)["status"] == "killed"


def test_a_second_rust_daemon_steps_aside(ts):
    task(ts, "rust", "daemon", "start")
    pid, _ = daemon(ts)
    (ts.state / "daemon.pid").unlink()              # as if the PID file were lost
    out = task(ts, "rust", "daemon", "start").stdout
    assert "Daemon started" in out
    deadline = time.time() + 10
    while int((ts.state / "daemon.pid").read_text()) != pid:
        assert time.time() < deadline, "the second daemon did not hand the PID file back"
        time.sleep(0.1)
    assert alive(pid)
    assert "Another daemon (PID %d) is already running; this one exits." % pid in \
        (ts.state / "daemon.log").read_text()
    assert task(ts, "rust", "daemon", "status").stdout.startswith(f"Daemon running (PID {pid})")
