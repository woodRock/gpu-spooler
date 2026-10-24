import sqlite3
import sys

import pytest

from gpu_spooler import config as C
from gpu_spooler.idle import fmt_idle, idle_gpus

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# The subset of gpu-detective's schema that gpu-spooler reads.
SCHEMA = """
CREATE TABLE snapshots (id INTEGER PRIMARY KEY, ts REAL, host TEXT, reachable INTEGER, error TEXT DEFAULT '');
CREATE TABLE gpus (snapshot_id INTEGER, idx INTEGER, name TEXT, mem_used_mb INTEGER,
                   mem_free_mb INTEGER, mem_total_mb INTEGER, util_pct INTEGER);
CREATE TABLE procs (snapshot_id INTEGER, gpu_idx INTEGER, pid INTEGER, user TEXT,
                    mem_mb INTEGER, started REAL, command TEXT);
"""


def history(path, sweeps):
    """sweeps: [(ts, host, reachable, {gpu: (util, [users])})]"""
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    for ts, host, reachable, gpus in sweeps:
        sid = con.execute("INSERT INTO snapshots (ts, host, reachable) VALUES (?, ?, ?)",
                          (ts, host, int(reachable))).lastrowid
        for idx, (util, users) in gpus.items():
            con.execute("INSERT INTO gpus VALUES (?, ?, 'A40', 1, 1, 2, ?)", (sid, idx, util))
            for i, u in enumerate(users):
                con.execute("INSERT INTO procs VALUES (?, ?, ?, ?, 1, 0, 'x')", (sid, idx, 100 + i, u))
    con.commit()
    con.close()


def test_idle_gpus(tmp_path):
    db = tmp_path / "h.db"
    now = 100000.0
    history(db, [
        (now - 3600, "a", True, {0: (90, ["alice"]), 1: (0, ["bob"])}),
        (now - 2400, "a", True, {0: (0, ["alice"]), 1: (0, ["bob"])}),
        (now - 1200, "a", True, {0: (0, ["alice"]), 1: (0, ["bob"])}),
        (now - 60, "a", True, {0: (0, ["alice"]), 1: (0, ["bob"]), 2: (0, [])}),
        (now - 60, "b", True, {0: (0, ["carol"])}),           # one sweep only: too short
        (now - 4000, "c", True, {0: (0, ["dave"])}),          # stale host
    ])
    idle = idle_gpus(db, 30 * 60, now=now)
    assert set(idle) == {("a", 0), ("a", 1)}
    assert idle[("a", 0)].idle_seconds == 2400 - 60 and not idle[("a", 0)].capped
    assert idle[("a", 1)].capped and fmt_idle(idle[("a", 1)]) == "idle 59m+"
    assert idle[("a", 0)].users == ["alice"]


def test_idle_breaks_on_new_holder_or_outage(tmp_path):
    db = tmp_path / "h.db"
    now = 100000.0
    history(db, [
        (now - 3000, "a", True, {0: (0, ["alice"]), 1: (0, ["bob"])}),
        (now - 2000, "a", False, {}),
        (now - 1000, "a", True, {0: (0, ["erin"]), 1: (0, ["bob"])}),
        (now - 60, "a", True, {0: (0, ["erin"]), 1: (0, ["bob"])}),
    ])
    assert idle_gpus(db, 600, now=now).keys() == {("a", 0), ("a", 1)}
    assert idle_gpus(db, 1200, now=now) == {}


def test_missing_or_bad_history_is_empty(tmp_path):
    assert idle_gpus(tmp_path / "nope.db", 60) == {}
    (tmp_path / "junk.db").write_text("not sqlite")
    assert idle_gpus(tmp_path / "junk.db", 60) == {}


def test_detective_config(monkeypatch):
    monkeypatch.delenv("GPU_DETECTIVE_DB", raising=False)
    off = C.parse_config({})
    assert off.detective_path() is None
    on = C.parse_config(tomllib.loads("[detective]\nidle_minutes = 45\n"))
    assert str(on.detective_path()).endswith(".gpu-detective/history.db") and on.idle_minutes == 45
    assert C.parse_config(tomllib.loads("[detective]\nenabled = false\n")).detective_path() is None
    with pytest.raises(C.ConfigError):
        C.parse_config(tomllib.loads("[detective]\nidle = 3\n"))
    with pytest.raises(C.ConfigError):
        C.parse_config(tomllib.loads("[detective]\nidle_minutes = 0\n"))


def test_confirm_idle_drops_what_the_live_query_contradicts():
    from gpu_spooler.gpu import GpuInfo, GpuProcess, ServerResult
    from gpu_spooler.gpu_status import confirm_idle
    from gpu_spooler.idle import IdleGpu

    def g(idx, util, user):
        return GpuInfo(idx, "A40", 9000, 0, 46068, util, [GpuProcess(1, user, 9000, "x")])
    results = [ServerResult("a", True, gpus=[g(0, 0, "alice"), g(1, 68, "bob"), g(2, 0, "erin")])]
    idle = {("a", i): IdleGpu("a", i, 3600, [u]) for i, u in enumerate(["alice", "bob", "carol"])}
    idle[("b", 0)] = IdleGpu("b", 0, 3600, ["dave"])     # host not queried this time
    assert set(confirm_idle(idle, results)) == {("a", 0)}
    assert confirm_idle(None, results) is None


def test_task_list_flags_only_fully_idle_running_tasks(monkeypatch):
    from gpu_spooler import gpu_status, task
    from gpu_spooler.idle import IdleGpu
    monkeypatch.setattr(gpu_status, "load_idle", lambda cfg: {
        ("a", 0): IdleGpu("a", 0, 3900, ["me"]), ("a", 1): IdleGpu("a", 1, 7200, ["me"])})
    monkeypatch.setattr(task, "get_cfg", lambda: C.parse_config({}))
    rows = [{"id": 1, "status": "running", "server": "a", "gpu_indices": "0,1"},
            {"id": 2, "status": "running", "server": "a", "gpu_indices": "1,2"},
            {"id": 3, "status": "done", "server": "a", "gpu_indices": "0"}]
    assert task.idle_tasks(rows) == {1: "idle 1h05m"}
