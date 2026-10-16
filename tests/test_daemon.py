"""Integration tests against dev/fake_cluster.sh (items 4-10 of the brief)."""
import subprocess
import time

import pytest

from conftest import fake

pytestmark = pytest.mark.integration


def log_body(ts, tid) -> str:
    return (ts.state / "logs" / f"{tid}.log").read_text()


def pgrep(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def test_submit_runs_to_done_and_syncs_log(ts):
    tid = ts.submit("-G", "1", "echo hi; sleep 3; echo bye")
    r = ts.wait_status(tid, "running", 30)
    assert r["server"] in ("cuda01", "cuda02") and r["job_key"]
    r = ts.wait_status(tid, ("done", "failed", "lost"), 30)
    assert r["status"] == "done" and r["exit_code"] == 0
    body = log_body(ts, tid)
    assert "hi\n" in body and "bye\n" in body
    out = ts.run("-l").stdout
    assert "network: jump (forced)" in out


def test_exit_code_retries_then_fails(ts):
    tid = ts.submit("-G", "1", "echo attempt; exit 3")
    r = ts.wait_status(tid, ("failed", "done", "lost"), 60)
    assert r["status"] == "failed" and r["exit_code"] == 3 and r["retry_count"] == 3


def test_readoption_after_daemon_restart(ts):
    tid = ts.submit("-G", "1", "sleep 8; echo finished")
    ts.wait_status(tid, "running", 30)
    ts.run("--stop")
    time.sleep(2)
    assert ts.row(tid)["status"] == "running"
    ts.run("--daemon")
    time.sleep(2)
    assert ts.row(tid)["status"] == "running"
    r = ts.wait_status(tid, ("done", "failed", "lost"), 30)
    assert r["status"] == "done"
    assert "finished" in log_body(ts, tid)


def test_kill_removes_process_group(ts):
    tid = ts.submit("-G", "1", "sleep 3017")
    r = ts.wait_status(tid, "running", 30)
    deadline = time.time() + 15
    while not (r["remote_pid"] and pgrep("sleep 3017")):
        assert time.time() < deadline, "job never started"
        time.sleep(0.3)
        r = ts.row(tid)
    out = ts.run("-k", str(tid)).stdout
    assert "Sent SIGTERM to process group" in out
    deadline = time.time() + 5
    while pgrep("sleep 3017") or pgrep(r["job_key"]):
        assert time.time() < deadline, "process group still alive"
        time.sleep(0.2)
    time.sleep(2)                                   # daemon must not retry it
    assert ts.row(tid)["status"] == "killed"


def test_connection_drop_keeps_job_running(ts):
    tid = ts.submit("-G", "1", "-H", "cuda01",
                    "for i in 1 2 3 4 5 6 7 8 9 10; do echo line$i; sleep 1; done")
    ts.wait_status(tid, "running", 30)
    time.sleep(1)
    fake("stop", "barretts")
    try:
        time.sleep(5)
        assert ts.row(tid)["status"] == "running"
    finally:
        fake("start", "barretts")
    r = ts.wait_status(tid, ("done", "failed", "lost"), 60)
    assert r["status"] == "done"
    body = log_body(ts, tid)
    assert all(f"line{i}\n" in body for i in range(1, 11)), body


def test_host_pinning_cpu_jobs_and_busy_gpus(ts):
    fake("busy", "cuda01", "0")
    try:
        big = ts.submit("-G", "1", "-m", "40", "echo big")
        pinned = ts.submit("-G", "1", "-H", "cuda01", "sleep 6")
        cpu = ts.submit("-G", "0", "echo cpu-$FAKE_HOST-$CUDA_VISIBLE_DEVICES-")
        r = ts.wait_status(pinned, "running", 30)
        assert r["server"] == "cuda01" and r["gpu_indices"] == "1"
        second = ts.submit("-G", "1", "-H", "cuda01", "echo second")
        time.sleep(3)
        assert ts.row(second)["status"] == "queued"   # GPU 0 busy, GPU 1 ours
        assert ts.wait_status(big, "done", 30)["server"] == "cuda02"
        r = ts.wait_status(cpu, "done", 30)
        assert r["server"] == "local" and "cpu---\n" in log_body(ts, cpu)
        r = ts.wait_status(second, "done", 40)
        assert r["server"] == "cuda01" and r["gpu_indices"] == "1"
    finally:
        fake("idle", "cuda01")


def test_bad_host_spec_rejected(ts):
    r = ts.run("-G", "1", "-H", "nosuchhost", "true", check=False)
    assert r.returncode != 0 and "nosuchhost" in r.stderr


def test_path_map_applied(ts, fake_cluster):
    src = fake_cluster / "local-src" / "proj"
    (fake_cluster / "remote-src" / "proj").mkdir(parents=True, exist_ok=True)
    src.mkdir(parents=True, exist_ok=True)
    tid = ts.submit("-G", "1", "pwd", cwd=str(src))
    r = ts.wait_status(tid, ("done", "failed", "lost"), 30)
    assert r["status"] == "done"
    assert r["remote_dir"] == str(fake_cluster / "remote-src" / "proj")
    assert str(fake_cluster / "remote-src" / "proj") in log_body(ts, tid)


def test_large_log_syncs_over_several_ticks(ts):
    tid = ts.submit("-G", "1", "yes 2>/dev/null | head -c 3000000")
    r = ts.wait_status(tid, ("done", "failed", "lost"), 60)
    assert r["status"] == "done" and r["log_offset"] == 3000000
    body = log_body(ts, tid)
    payload = body.split("─" * 60 + "\n", 1)[1]
    assert len(payload) == 3000000 and set(payload) == {"y", "\n"}


def test_clear_removes_remote_job_dirs(ts, fake_cluster):
    tid = ts.submit("-G", "1", "true")
    r = ts.wait_status(tid, "done", 30)
    jobdir = fake_cluster / "home" / ".task-spooler" / "jobs" / r["job_key"]
    assert jobdir.is_dir()
    ts.run("-c")
    assert not jobdir.exists()


def test_old_running_rows_marked_lost(ts):
    import sqlite3
    ts.run("-l")                                    # creates the db
    with sqlite3.connect(ts.state / "tasks.db") as con:
        con.execute("INSERT INTO tasks (command, status, submitted_at, working_dir, ssh_pid) "
                    "VALUES ('old', 'running', 1, '/tmp', 99999)")
    ts.run("--daemon")
    assert ts.wait_status(1, "lost", 15)["exit_code"] is None


def test_kill_all_in_parallel(ts):
    a = ts.submit("-G", "1", "sleep 3018")
    b = ts.submit("-G", "0", "sleep 3019")
    ts.wait_status(a, "running", 30)
    ts.wait_status(b, "running", 30)
    deadline = time.time() + 15
    while not (pgrep("sleep 3018") and pgrep("sleep 3019")):
        assert time.time() < deadline
        time.sleep(0.3)
    out = ts.run("-K").stdout
    assert "killed 2 running task(s)" in out
    deadline = time.time() + 5
    while pgrep("sleep 3018") or pgrep("sleep 3019"):
        assert time.time() < deadline, "processes survived -K"
        time.sleep(0.2)


def test_daemon_restarts_itself_when_code_changes(ts):
    import os
    from pathlib import Path
    tid = ts.submit("-G", "1", "sleep 6; echo survived")
    ts.wait_status(tid, "running", 30)
    pid = int((ts.state / "daemon.pid").read_text())
    src = Path(__file__).resolve().parent.parent / "gpu_spooler" / "jobs.py"
    os.utime(src)                                     # looks like an update
    log = ts.state / "daemon.log"
    deadline = time.time() + 15
    while "Code changed on disk" not in log.read_text():
        assert time.time() < deadline, log.read_text()[-2000:]
        time.sleep(0.3)
    os.kill(pid, 0)                                   # same process, re-exec'd
    r = ts.wait_status(tid, ("done", "failed", "lost"), 30)
    assert r["status"] == "done" and "survived" in log_body(ts, tid)
