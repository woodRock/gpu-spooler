"""The kind = "sge" backend, against the mock qsub, qstat and qdel in dev/fake_cluster.sh."""
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def sge(fake_cluster):
    flags = [fake_cluster / "sge" / f for f in ("hold", "down")]
    yield fake_cluster / "sge"
    for f in flags:
        f.unlink(missing_ok=True)


def log_body(ts, tid) -> str:
    return (ts.state / "logs" / f"{tid}.log").read_text()


def wait_for(cond, timeout=30, what="condition"):
    deadline = time.time() + timeout
    while not cond():
        assert time.time() < deadline, f"timed out waiting for {what}"
        time.sleep(0.3)


def test_job_waits_runs_and_finishes(ts, sge):
    (sge / "hold").touch()
    tid = ts.submit("submit", "-g", "0", "-H", "grid", 'echo "scratch=$TASK_SCRATCH"; pwd; exit 3')
    ts.wait_status(tid, "running", 30)
    wait_for(lambda: ts.row(tid)["sge_job_id"], what="the SGE job ID")
    r = ts.row(tid)
    assert r["server"] == "grid" and r["sge_state"] == "qw" and r["started_at"] is None
    wait_for(lambda: "waiting" in ts.run("list").stdout, what="waiting in task list")
    assert "SGE job" in ts.run("info", str(tid)).stdout

    (sge / "hold").unlink()
    r = ts.wait_status(tid, ("done", "failed", "lost"), 30)
    assert r["status"] == "failed" and r["exit_code"] == 3
    assert r["started_at"] is not None
    assert r["finished_at"] - r["started_at"] < 2  # not when a poll noticed
    assert f"/{r['sge_job_id']}\n" in log_body(ts, tid).replace("scratch=/local/tmp/", "")


def test_kill_deletes_the_job(ts, sge):
    (sge / "hold").touch()
    tid = ts.submit("submit", "-g", "0", "-H", "grid", "sleep 300")
    ts.wait_status(tid, "running", 30)
    wait_for(lambda: ts.row(tid)["sge_job_id"], what="the SGE job ID")
    jid = ts.row(tid)["sge_job_id"]
    assert (sge / "jobs" / str(jid)).is_dir()
    assert f"Deleted job {jid} from SGE" in ts.run("kill", str(tid)).stdout
    assert not (sge / "jobs" / str(jid)).exists()
    assert ts.row(tid)["status"] == "killed"


def test_killing_a_job_that_just_left_qstat(ts, sge):
    (sge / "hold").touch()
    tid = ts.submit("submit", "-g", "0", "-H", "grid", "sleep 300")
    ts.wait_status(tid, "running", 30)
    wait_for(lambda: ts.row(tid)["sge_job_id"], what="the SGE job ID")
    shutil.rmtree(sge / "jobs" / str(ts.row(tid)["sge_job_id"]))
    assert "already gone" in ts.run("kill", str(tid)).stdout


def test_qstat_failing_does_not_lose_jobs(ts, sge):
    (sge / "hold").touch()
    tid = ts.submit("submit", "-g", "0", "-H", "grid", "echo survived")
    ts.wait_status(tid, "running", 30)
    (sge / "down").touch()
    time.sleep(8)
    assert ts.row(tid)["status"] == "running"
    (sge / "down").unlink()
    (sge / "hold").unlink()
    r = ts.wait_status(tid, ("done", "failed", "lost"), 30)
    assert r["status"] == "done"
    assert "survived" in log_body(ts, tid)


def test_a_job_that_vanishes_is_lost_after_a_grace_period(ts, sge):
    (sge / "hold").touch()
    tid = ts.submit("submit", "-g", "0", "-H", "grid", "sleep 300")
    ts.wait_status(tid, "running", 30)
    wait_for(lambda: ts.row(tid)["sge_job_id"], what="the SGE job ID")
    jid = ts.row(tid)["sge_job_id"]
    shutil.rmtree(sge / "jobs" / str(jid))
    time.sleep(4)
    assert ts.row(tid)["status"] == "running"
    assert ts.wait_status(tid, ("lost", "done", "failed"), 40)["status"] == "lost"


def test_hosts_reports_the_submit_host(ts, sge):
    line = next(l for l in ts.run("hosts").stdout.splitlines() if l.startswith("grid "))
    assert " sge " in line and "ok  " in line and "unreachable" not in line


def test_gpu_jobs_are_refused(ts, sge):
    r = ts.run("submit", "-g", "1", "-H", "grid", "true", check=False)
    assert r.returncode != 0 and "use -g 0" in (r.stdout + r.stderr)


def test_unnamed_jobs_stay_off_the_grid(ts, sge):
    assert "grid" not in ts.run("gpus").stdout
    tid = ts.submit("submit", "-g", "0", "echo here")
    assert ts.wait_status(tid, ("done", "failed", "lost"), 30)["server"] == "local"


# ── run records: taura's own view of them, the same as for a GPU task ────────────

GIT = ["git", "-c", "user.email=t@example.com", "-c", "user.name=t"]


@pytest.fixture
def repo():
    root = Path(tempfile.mkdtemp(prefix="tsp.", dir="/tmp")) / "proj"
    (root / "ledger").mkdir(parents=True)
    (root / "ledger" / "artifacts.toml").write_text("")
    (root / "train.py").write_text("print('v1')\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    commit(root)
    yield root
    shutil.rmtree(root.parent, ignore_errors=True)


def commit(root) -> str:
    subprocess.run([*GIT, "-C", str(root), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(root), "commit", "-qm", "change"], check=True)
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def runs(root):
    taura = shutil.which("taura") or pytest.skip("taura is not installed")
    r = subprocess.run([taura, "-C", str(root), "--json", "runs"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)["runs"]


def test_grid_task_is_recorded_from_when_it_runs(ts, sge, repo):
    runs(repo)
    ts.run("sync", cwd=str(repo))
    (sge / "hold").touch()
    tid = ts.submit("submit", "-g", "0", "-H", "grid", "python3 train.py --seed 5", cwd=str(repo))
    wait_for(lambda: ts.row(tid)["sge_job_id"], what="the SGE job ID")
    assert runs(repo) == []  # queued: it hasn't run

    (repo / "train.py").write_text("print('v2')\n")
    head = commit(repo)
    ts.run("sync", cwd=str(repo))
    time.sleep(2)  # submitted well before it starts
    (sge / "hold").unlink()

    r = ts.wait_status(tid, ("done", "failed", "lost"), 30)
    assert r["status"] == "done"
    [run] = runs(repo)
    assert run["commit"] == head and run["dirty"] is False  # what it ran, not what was there at qsub
    assert run["status"] == "done" and run["exit_code"] == 0
    assert run["spooler_job"] == tid and run["runner"] == "gpu-spooler" and run["seed"] == 5
    assert run["host"] == r["machine"] != "grid"
    assert run["gpus"] == ""
    assert run["inputs"] == ["train.py"]
    assert run["env"]["python"] and run["env"]["hash"].startswith("sha256:")
    assert r["finished_at"] - r["started_at"] < 2 and r["started_at"] > r["submitted_at"] + 2
    assert datetime.fromisoformat(run["started"]).timestamp() == r["started_at"]
    assert datetime.fromisoformat(run["finished"]).timestamp() == r["finished_at"]
    head_lines = log_body(ts, tid).split("─" * 60)[0]
    assert "# server: grid" in head_lines and f"# machine: {r['machine']}\n" in head_lines
    assert "# submitted: " in head_lines and "# started: " in head_lines


def test_failed_grid_task_is_recorded(ts, sge, repo):
    ts.run("sync", cwd=str(repo))
    tid = ts.submit("submit", "-g", "0", "-H", "grid", "exit 3", cwd=str(repo))
    assert ts.wait_status(tid, ("done", "failed", "lost"), 30)["status"] == "failed"
    [run] = runs(repo)
    assert run["status"] == "failed" and run["exit_code"] == 3 and run["finished"]


def test_grid_task_lost_after_it_started_is_recorded(ts, sge, repo):
    ts.run("sync", cwd=str(repo))
    tid = ts.submit("submit", "-g", "0", "-H", "grid", "sleep 300", cwd=str(repo))
    wait_for(lambda: ts.row(tid)["started_at"], what="the job to start")
    wait_for(lambda: runs(repo), what="the start record")
    assert runs(repo)[0]["status"] == "running"
    jid = str(ts.row(tid)["sge_job_id"])
    os.killpg(int((sge / "jobs" / jid / "pid").read_text()), signal.SIGKILL)
    shutil.rmtree(sge / "jobs" / jid)
    r = ts.wait_status(tid, ("lost", "done", "failed"), 40)
    assert r["status"] == "lost" and r["finished_at"] < time.time() - 10  # not the end of the grace period
    [run] = runs(repo)
    assert run["status"] == "failed" and "exit_code" not in run and run["note"].startswith("Lost:")
    assert run["host"] == r["machine"]


def test_grid_task_lost_in_the_queue_is_recorded(ts, sge, repo):
    ts.run("sync", cwd=str(repo))
    (sge / "hold").touch()
    tid = ts.submit("submit", "-g", "0", "-H", "grid", "python3 train.py", cwd=str(repo))
    wait_for(lambda: ts.row(tid)["sge_job_id"], what="the SGE job ID")
    shutil.rmtree(sge / "jobs" / str(ts.row(tid)["sge_job_id"]))
    assert ts.wait_status(tid, ("lost", "done", "failed"), 40)["status"] == "lost"
    [run] = runs(repo)
    assert run["status"] == "failed" and run["host"] == "grid" and "never ran" in run["note"]
    assert run["command"] == "python3 train.py" and run["started"] == run["finished"]


def test_no_record_grid_task_writes_nothing(ts, sge, repo):
    ts.run("sync", cwd=str(repo))
    tid = ts.submit("submit", "--no-record", "-g", "0", "-H", "grid", "echo hi", cwd=str(repo))
    assert ts.wait_status(tid, ("done", "failed", "lost"), 30)["status"] == "done"
    assert runs(repo) == []
    assert "TAURA_ENV" not in log_body(ts, tid)


def test_a_daemon_started_with_launchds_path_still_finds_taura(ts, sge, repo):
    taura = shutil.which("taura") or pytest.skip("taura is not installed")
    (ts.home / ".local" / "bin").mkdir(parents=True)
    (ts.home / ".local" / "bin" / "taura").symlink_to(taura)
    ts.env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    ts.run("sync", cwd=str(repo))
    tid = ts.submit("submit", "-g", "0", "-H", "grid", "echo hi", cwd=str(repo))
    assert ts.wait_status(tid, ("done", "failed", "lost"), 30)["status"] == "done"
    assert runs(repo)[0]["env"]["python"]


def test_a_daemon_without_taura_says_so(ts, sge, repo):
    ts.env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    if shutil.which("taura", path=ts.env["PATH"]):
        pytest.skip("taura is on the system PATH")
    ts.run("sync", cwd=str(repo))
    tid = ts.submit("submit", "-g", "0", "-H", "grid", "echo hi", cwd=str(repo))
    assert ts.wait_status(tid, ("done", "failed", "lost"), 30)["status"] == "done"
    wait_for(lambda: "taura was not found" in (ts.state / "daemon.log").read_text(), what="the warning")
