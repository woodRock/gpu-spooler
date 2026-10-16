import base64
import os
import sqlite3
import subprocess
import time

from gpu_spooler import jobs
from gpu_spooler.config import parse_config
from gpu_spooler.gpu import GpuInfo, ServerResult


def sh(script: str, home) -> str:
    return subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True,
                          env=dict(os.environ, HOME=str(home)), cwd=str(home),
                          timeout=30).stdout


def wait_rc(home, key, timeout=10):
    rc = home / ".task-spooler" / "jobs" / key / "rc"
    deadline = time.time() + timeout
    while not rc.exists():
        assert time.time() < deadline, "job did not finish"
        time.sleep(0.1)
    return int(rc.read_text())


def test_launch_exit_in_subshell_and_quoting(tmp_path):
    wdir = tmp_path / "dir with 'quotes' $x"
    wdir.mkdir()
    key = jobs.new_job_key(7)
    cmd = "echo \"gpus=$CUDA_VISIBLE_DEVICES\"; pwd; echo TS_EOF_x; exit 3  # trailing comment"
    out = sh(jobs.launch_script(key, str(wdir), "1,2", cmd, delay=False), tmp_path)
    pid = jobs.parse_launch(out)
    assert pid
    assert wait_rc(tmp_path, key) == 3
    log = (tmp_path / ".task-spooler" / "jobs" / key / "out.log").read_text()
    assert "gpus=1,2" in log and str(wdir) in log and "TS_EOF_x" in log


def test_launch_bad_dir_writes_rc(tmp_path):
    key = jobs.new_job_key(1)
    sh(jobs.launch_script(key, str(tmp_path / "missing"), "", "true", delay=False), tmp_path)
    assert wait_rc(tmp_path, key) == 1
    assert "cannot cd" in (tmp_path / ".task-spooler/jobs" / key / "out.log").read_text()


def test_status_offsets_and_states(tmp_path):
    key = jobs.new_job_key(2)
    sh(jobs.launch_script(key, str(tmp_path), "", "printf 'abcdef'; sleep 30", delay=False),
       tmp_path)
    time.sleep(0.5)
    st = jobs.parse_status(sh(jobs.status_script([(key, 2), ("9-nothere", 0)]), tmp_path))
    assert st[key].state == "running" and st[key].size == 6 and st[key].data == b"cdef"
    assert st["9-nothere"].state == "missing"
    out = sh(jobs.kill_script(key, grace=1), tmp_path)
    assert "KILLED" in out
    time.sleep(0.5)
    st = jobs.parse_status(sh(jobs.status_script([(key, 6)]), tmp_path))
    assert st[key].state == "gone" and st[key].data == b""
    assert "NOT_RUNNING" in sh(jobs.kill_script(key), tmp_path)
    sh(jobs.cleanup_script([2]), tmp_path)
    assert not (tmp_path / ".task-spooler/jobs" / key).exists()


def test_parse_status_truncated():
    data = base64.b64encode(b"x").decode()
    assert jobs.parse_status(f"@@JOB 1-a\nSTATE done 0\nSIZE 1\nDATA {data}\n") is None
    st = jobs.parse_status(f"@@JOB 1-a\nSTATE done 0\nSIZE 1\nDATA {data}\n@@END\n")
    assert st["1-a"].rc == 0 and st["1-a"].data == b"x"


# ── assignment ───────────────────────────────────────────────────────────────

def _cfg():
    import tomllib
    return parse_config(tomllib.loads('''
[groups.g]
hosts = ["small", "big"]
[hosts.big]
gpu_mem_gb = 48
[hosts.loc]
kind = "local"
cpu_slots = 1
'''))


def _gpus(n, free=None, mib=24564):
    free = free if free is not None else [True] * n
    return [GpuInfo(i, "X", 1 if free[i] else 9000, 0, mib, 0) for i in range(n)]


def _row(id, n, mem=0, spec=None):
    return {"id": id, "num_gpus": n, "min_mem": mem, "host_spec": spec}


def test_assign_bin_packs_and_filters():
    from gpu_spooler.task import assign_tasks, host_states
    cfg = _cfg()
    results = {
        "small": ServerResult("small", True, gpus=_gpus(2, [False, True])),
        "big": ServerResult("big", True, gpus=_gpus(4, mib=46068)),
        "loc": ServerResult("loc", True),
    }
    running = [{"server": "big", "gpu_indices": "3", "num_gpus": 1}]
    states = host_states(cfg, results, running)
    rows = [_row(1, 1), _row(2, 1), _row(3, 1, mem=40), _row(4, 0), _row(5, 0),
            _row(6, 3, spec="big"), _row(7, 1, spec="loc")]
    got = assign_tasks(rows, states, cfg)
    assert got == [
        (1, "small", [1]),       # fewest GPUs first, busy GPU 0 skipped
        (2, "big", [0]),
        (3, "big", [1]),         # min_mem uses configured gpu_mem_gb
        (4, "loc", []),          # CPU slot
    ]                            # 5: no CPU slot left; 6: only GPU 2 left; 7: loc has no GPUs


def test_assign_min_mem_uses_detected_memory():
    from gpu_spooler.task import assign_tasks, host_states
    cfg = _cfg()
    results = {"small": ServerResult("small", True, gpus=_gpus(2, mib=46068))}
    states = host_states(cfg, results, [])
    assert assign_tasks([_row(1, 1, mem=48)], states, cfg) == [(1, "small", [0])]
    states = host_states(cfg, results, [])
    assert assign_tasks([_row(1, 1, mem=49)], states, cfg) == []


def test_db_migration_from_old_schema(tmp_path, monkeypatch):
    from gpu_spooler import task
    db = tmp_path / "tasks.db"
    con = sqlite3.connect(db)
    con.executescript('''CREATE TABLE tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT, command TEXT NOT NULL,
        num_gpus INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL DEFAULT 'queued',
        server TEXT, gpu_indices TEXT, ssh_pid INTEGER, submitted_at REAL NOT NULL,
        started_at REAL, finished_at REAL, exit_code INTEGER, log_file TEXT,
        working_dir TEXT NOT NULL, name TEXT);
        INSERT INTO tasks (command, status, submitted_at, working_dir, ssh_pid)
        VALUES ('echo', 'running', 1, '/tmp', 123);''')
    con.commit()
    con.close()
    monkeypatch.setattr(task, "DB_PATH", db)
    monkeypatch.setattr(task, "STATE_DIR", tmp_path)
    monkeypatch.setattr(task, "LOGS_DIR", tmp_path / "logs")
    r = dict(task.db_open().execute("SELECT * FROM tasks").fetchone())
    assert r["ssh_pid"] == 123 and r["job_key"] is None and r["log_offset"] == 0
    assert {"retry_count", "min_mem", "remote_pid", "host_spec", "remote_dir",
            "last_seen"} <= set(r)
