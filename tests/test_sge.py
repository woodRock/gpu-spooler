"""The kind = "sge" backend, against the mock qsub, qstat and qdel in dev/fake_cluster.sh."""
import shutil
import time

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
