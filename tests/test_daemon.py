"""Integration tests against dev/fake_cluster.sh (items 4-10 of the brief)."""
import subprocess
import sys
import time

import pytest

from conftest import fake

pytestmark = pytest.mark.integration


def log_body(ts, tid) -> str:
    return (ts.state / "logs" / f"{tid}.log").read_text()


def pgrep(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def test_submit_runs_to_done_and_syncs_log(ts):
    tid = ts.submit("submit", "-g", "1", "echo hi; sleep 3; echo bye")
    r = ts.wait_status(tid, "running", 30)
    assert r["server"] in ("cuda01", "cuda02") and r["job_key"]
    r = ts.wait_status(tid, ("done", "failed", "lost"), 30)
    assert r["status"] == "done" and r["exit_code"] == 0
    body = log_body(ts, tid)
    assert "hi\n" in body and "bye\n" in body
    out = ts.run("list").stdout
    assert "network: jump (forced)" in out


def test_exit_code_retries_then_fails(ts):
    tid = ts.submit("submit", "-g", "1", "echo attempt; exit 3")
    r = ts.wait_status(tid, ("failed", "done", "lost"), 60)
    assert r["status"] == "failed" and r["exit_code"] == 3 and r["retry_count"] == 3


def test_readoption_after_daemon_restart(ts):
    tid = ts.submit("submit", "-g", "1", "sleep 8; echo finished")
    ts.wait_status(tid, "running", 30)
    ts.run("daemon", "stop")
    time.sleep(2)
    assert ts.row(tid)["status"] == "running"
    ts.run("daemon", "start")
    time.sleep(2)
    assert ts.row(tid)["status"] == "running"
    r = ts.wait_status(tid, ("done", "failed", "lost"), 30)
    assert r["status"] == "done"
    assert "finished" in log_body(ts, tid)


def test_kill_removes_process_group(ts):
    tid = ts.submit("submit", "-g", "1", "sleep 3017")
    r = ts.wait_status(tid, "running", 30)
    deadline = time.time() + 15
    while not (r["remote_pid"] and pgrep("sleep 3017")):
        assert time.time() < deadline, "job never started"
        time.sleep(0.3)
        r = ts.row(tid)
    out = ts.run("kill", str(tid)).stdout
    assert "Sent SIGTERM to process group" in out
    deadline = time.time() + 5
    while pgrep("sleep 3017") or pgrep(r["job_key"]):
        assert time.time() < deadline, "process group still alive"
        time.sleep(0.2)
    time.sleep(2)                                   # daemon must not retry it
    assert ts.row(tid)["status"] == "killed"


def test_connection_drop_keeps_job_running(ts):
    tid = ts.submit("submit", "-g", "1", "-H", "cuda01",
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
        big = ts.submit("submit", "-g", "1", "-m", "40", "echo big")
        pinned = ts.submit("submit", "-g", "1", "-H", "cuda01", "sleep 6")
        cpu = ts.submit("submit", "-g", "0", "echo cpu-$FAKE_HOST-$CUDA_VISIBLE_DEVICES-")
        r = ts.wait_status(pinned, "running", 30)
        assert r["server"] == "cuda01" and r["gpu_indices"] == "1"
        second = ts.submit("submit", "-g", "1", "-H", "cuda01", "echo second")
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
    r = ts.run("submit", "-g", "1", "-H", "nosuchhost", "true", check=False)
    assert r.returncode != 0 and "nosuchhost" in r.stderr


def test_path_map_applied(ts, fake_cluster):
    src = fake_cluster / "local-src" / "proj"
    (fake_cluster / "remote-src" / "proj").mkdir(parents=True, exist_ok=True)
    src.mkdir(parents=True, exist_ok=True)
    tid = ts.submit("submit", "-g", "1", "pwd", cwd=str(src))
    r = ts.wait_status(tid, ("done", "failed", "lost"), 30)
    assert r["status"] == "done"
    assert r["remote_dir"] == str(fake_cluster / "remote-src" / "proj")
    assert str(fake_cluster / "remote-src" / "proj") in log_body(ts, tid)


def test_large_log_syncs_over_several_ticks(ts):
    tid = ts.submit("submit", "-g", "1", "yes 2>/dev/null | head -c 3000000")
    r = ts.wait_status(tid, ("done", "failed", "lost"), 60)
    assert r["status"] == "done" and r["log_offset"] == 3000000
    body = log_body(ts, tid)
    payload = body.split("─" * 60 + "\n", 1)[1]
    assert len(payload) == 3000000 and set(payload) == {"y", "\n"}


def test_clear_removes_remote_job_dirs(ts, fake_cluster):
    tid = ts.submit("submit", "-g", "1", "true")
    r = ts.wait_status(tid, "done", 30)
    jobdir = fake_cluster / "home" / ".gpu-spooler" / "jobs" / r["job_key"]
    assert jobdir.is_dir()
    ts.run("clear")
    assert not jobdir.exists()


def test_old_running_rows_marked_lost(ts):
    import sqlite3
    ts.run("list")                                    # creates the db
    with sqlite3.connect(ts.state / "tasks.db") as con:
        con.execute("INSERT INTO tasks (command, status, submitted_at, working_dir, ssh_pid) "
                    "VALUES ('old', 'running', 1, '/tmp', 99999)")
    ts.run("daemon", "start")
    assert ts.wait_status(1, "lost", 15)["exit_code"] is None


def test_kill_all_in_parallel(ts):
    a = ts.submit("submit", "-g", "1", "sleep 3018")
    b = ts.submit("submit", "-g", "0", "sleep 3019")
    ts.wait_status(a, "running", 30)
    ts.wait_status(b, "running", 30)
    deadline = time.time() + 15
    while not (pgrep("sleep 3018") and pgrep("sleep 3019")):
        assert time.time() < deadline
        time.sleep(0.3)
    out = ts.run("kill", "--all").stdout
    assert "killed 2 running task(s)" in out
    deadline = time.time() + 5
    while pgrep("sleep 3018") or pgrep("sleep 3019"):
        assert time.time() < deadline, "processes survived kill --all"
        time.sleep(0.2)


def test_daemon_restarts_itself_when_code_changes(ts):
    import os
    from pathlib import Path
    tid = ts.submit("submit", "-g", "1", "sleep 6; echo survived")
    ts.wait_status(tid, "running", 30)
    pid = int((ts.state / "daemon.pid").read_text())
    from conftest import BIN
    os.utime(BIN)                                     # looks like an upgrade: the daemon watches its binary
    log = ts.state / "daemon.log"
    deadline = time.time() + 15
    while "Code changed on disk" not in log.read_text():
        assert time.time() < deadline, log.read_text()[-2000:]
        time.sleep(0.3)
    os.kill(pid, 0)                                   # same process, re-exec'd
    r = ts.wait_status(tid, ("done", "failed", "lost"), 30)
    assert r["status"] == "done" and "survived" in log_body(ts, tid)


# ── task -w exit codes ────────────────────────────────────────────────────────

def test_wait_exit_codes_done_and_failed(ts):
    ok = ts.submit("submit", "-g", "1", "exit 0")
    bad = ts.submit("submit", "-g", "1", "exit 3")
    r = ts.run("wait", str(ok), check=False, timeout=90)
    assert r.returncode == 0 and r.stdout == ""
    assert r.stderr.startswith(f"task {ok}: done (exit 0")
    r = ts.run("wait", str(bad), check=False, timeout=90)
    assert r.returncode == 3 and ts.row(bad)["retry_count"] == 3
    assert f"task {bad}: failed (exit 3" in r.stderr
    # Already finished: returns at once; several ids use the first non-zero code.
    start = time.time()
    r = ts.run("wait", str(bad), str(ok), check=False)
    assert r.returncode == 3 and time.time() - start < 5
    assert r.stdout == "" and "\x1b[" not in r.stdout + r.stderr


def test_wait_killed_timeout_and_unknown(ts):
    tid = ts.submit("submit", "-g", "1", "sleep 60")
    ts.wait_status(tid, "running", 30)
    start = time.time()
    r = ts.run("wait", str(tid), "--timeout", "2", check=False)
    assert r.returncode == 124 and 2 <= time.time() - start < 10
    assert "timed out" in r.stderr and r.stdout == ""
    assert ts.row(tid)["status"] == "running"

    import signal
    from conftest import task_argv
    p = subprocess.Popen(task_argv("wait", str(tid)),
                         env=ts.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(1.5)
    p.send_signal(signal.SIGINT)
    out, err = p.communicate(timeout=10)
    assert p.returncode == 130 and out == "" and "keep running" in err
    assert ts.row(tid)["status"] == "running"

    ts.run("kill", str(tid))
    r = ts.run("wait", str(tid), check=False)
    assert r.returncode == 130 and f"task {tid}: killed" in r.stderr

    q = ts.submit("submit", "-g", "1", "-H", "cuda01", "-m", "1000", "true")   # never fits
    ts.run("kill", str(q))
    assert ts.run("wait", str(q), check=False).returncode == 130       # cancelled

    r = ts.run("wait", "99999", check=False)
    assert r.returncode == 1 and "99999 not found" in r.stderr and r.stdout == ""


def test_wait_lost_returns_75(ts):
    import sqlite3
    tid = ts.submit("submit", "-g", "1", "-H", "cuda01", "-m", "1000", "true")
    with sqlite3.connect(ts.state / "tasks.db") as con:
        con.execute("UPDATE tasks SET status='lost' WHERE id=?", (tid,))
    assert ts.run("wait", str(tid), check=False).returncode == 75


def _queued_id(out: str) -> int:
    return int(next(l for l in out.splitlines() if l.startswith("Queued task ")).split()[2]
               .rstrip(":"))


def test_gpu_type_picks_hosts_by_their_live_gpu_name(ts):
    a40 = ts.submit("submit", "-g", "1", "-t", "A40", "echo a")
    a5k = ts.submit("submit", "-g", "1", "--gpu-type", "rtx a5000", "echo b")
    assert ts.wait_status(a40, "done", 40)["server"] == "cuda02"
    assert ts.wait_status(a5k, "done", 40)["server"] == "cuda01"
    info = ts.run("info", str(a5k)).stdout
    assert any("GPU type:" in l and "rtx a5000" in l for l in info.splitlines())
    assert any("Server:" in l and "cuda01" in l and "RTX A5000" in l for l in info.splitlines())
    out = ts.run("submit", "-g", "1", "--gpu-type", "L4", "true").stdout
    assert "no selected host reported a GPU of type 'L4'" in out
    tid = _queued_id(out)
    time.sleep(3)
    assert ts.row(tid)["status"] == "queued"
    ts.run("kill", str(tid))
    r = ts.run("submit", "-g", "0", "--gpu-type", "A40", "true", check=False)
    assert r.returncode != 0 and "--gpu-type needs a GPU job" in r.stderr


def test_max_gpus_caps_concurrent_gpus(fake_cluster):
    import shutil
    import tempfile
    from pathlib import Path

    from conftest import TS
    cfg = Path(tempfile.mkdtemp(prefix="tsc.", dir="/tmp")) / "hosts.toml"
    cfg.write_text("[limits]\nmax_gpus = 2\n\n" + (fake_cluster / "hosts.toml").read_text())
    env = TS(Path(tempfile.mkdtemp(prefix="tsh.", dir="/tmp")), cfg)
    try:
        r = env.run("submit", "-g", "3", "true", check=False)
        assert r.returncode != 0 and "max_gpus = 2" in r.stderr
        ids = [env.submit("submit", "-g", "1", "sleep 4") for _ in range(4)]   # 6 GPUs free
        peak, deadline = 0, time.time() + 90
        while time.time() < deadline:
            rows = [env.row(i) for i in ids]
            peak = max(peak, sum(r["status"] == "running" for r in rows))
            if all(r["status"] in ("done", "failed", "lost") for r in rows):
                break
            time.sleep(0.3)
        assert [env.row(i)["status"] for i in ids] == ["done"] * 4
        assert peak == 2
        assert "GPUs 0/2 (max_gpus)" in env.run("list").stdout
    finally:
        env.run("daemon", "stop", check=False)
        shutil.rmtree(env.home, ignore_errors=True)
