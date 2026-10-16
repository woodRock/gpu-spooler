import asyncio
import json
import time

import pytest

pytest.importorskip("textual")

from gpu_spooler import core, tui                      # noqa: E402
from gpu_spooler.config import builtin_config, parse_config   # noqa: E402

import tomllib                                          # noqa: E402


@pytest.fixture
def state(tmp_path, monkeypatch):
    """Point core at an empty state dir with a local-only config; no daemon."""
    for name, rel in [("STATE_DIR", ""), ("DB_PATH", "tasks.db"), ("PID_FILE", "daemon.pid"),
                      ("DAEMON_LOG", "daemon.log"), ("LOGS_DIR", "logs"),
                      ("SNAPSHOT_PATH", "gpu_snapshot.json"),
                      ("SNAPSHOT_REQUEST", "gpu_snapshot.request")]:
        monkeypatch.setattr(core, name, tmp_path / rel if rel else tmp_path)
    cfg = parse_config(tomllib.loads('[hosts.local]\nkind = "local"\ncpu_slots = 1\n'))
    core.set_config(cfg)
    monkeypatch.setattr(core, "start_daemon", lambda: None)
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    core.set_config(builtin_config())


def run_app(test):
    async def go():
        app = tui.SpoolerApp()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            await test(app, pilot)
    asyncio.run(go())


def test_table_shows_tasks_and_kill_cancels(state):
    a = core.submit("echo one", 1).task_id
    b = core.submit("echo two", 0, name="cpu job").task_id

    async def t(app, pilot):
        table = app.query_one(tui.DataTable)
        assert table.row_count == 2
        assert app.selected_id() == b              # newest first
        assert "1 queued" not in app.sub_title and "2 queued" in app.sub_title
        await pilot.press("k")
        await pilot.press("y")
        await app.workers.wait_for_complete()
        assert core.get_task(b)["status"] == "cancelled"
        assert core.get_task(a)["status"] == "queued"
        await pilot.press("down", "k", "n")         # "no" leaves it alone
        await pilot.pause()
        assert core.get_task(a)["status"] == "queued"
    run_app(t)


def test_resubmit_and_new_task_form(state):
    orig = core.submit("python3 x.py", 2, min_mem=40, name="lbl", host_spec="local")

    async def t(app, pilot):
        await pilot.press("r")
        await app.workers.wait_for_complete()
        copy = core.get_task(orig.task_id + 1)
        assert (copy["command"], copy["num_gpus"], copy["min_mem"], copy["name"],
                copy["host_spec"], copy["working_dir"]) == (
            "python3 x.py", 2, 40, "lbl", "local", orig.working_dir)

        await pilot.press("n")
        await pilot.pause()
        app.screen.query_one("#command").value = "echo hi && echo there"
        app.screen.query_one("#gpus").value = "0"
        app.screen.query_one("#name").value = "from tui"
        await pilot.click("#ok")
        await app.workers.wait_for_complete()
        new = core.get_task(orig.task_id + 2)
        assert (new["command"], new["num_gpus"], new["name"]) == (
            "echo hi && echo there", 0, "from tui")
    run_app(t)


def test_log_and_info_panes(state):
    sub = core.submit("echo hi", 1)
    log = core.LOGS_DIR / f"{sub.task_id}.log"
    log.write_text("# header\nfirst line\n")
    con = core.db_open()
    con.execute("UPDATE tasks SET status='running', server='local', started_at=?, log_file=? "
                "WHERE id=?", (time.time(), str(log), sub.task_id))
    con.commit()

    async def t(app, pilot):
        await pilot.pause(0.6)
        view = app.query_one(tui.Log)
        assert any("first line" in l for l in view.lines)
        with open(log, "a") as f:
            f.write("second line\n")
        await pilot.pause(1.2)
        assert any("second line" in l for l in view.lines)
        await pilot.press("f")
        assert app.screen.maximized is not None and app.query_one(tui.TabbedContent).active == "log"
        await pilot.press("f")
        assert app.screen.maximized is None
        await pilot.press("i")
        details = str(app.query_one("#details").render())
        assert "running" in details and "local" in details
    run_app(t)


def test_gpu_map_from_snapshot(state):
    sub = core.submit("train", 1)
    con = core.db_open()
    con.execute("UPDATE tasks SET status='running', server='cuda01', gpu_indices='1' WHERE id=?",
                (sub.task_id,))
    con.commit()
    gpu = lambda i, used, procs=(): {
        "index": i, "name": "RTX A5000", "mem_used_mb": used, "mem_free_mb": 0,
        "mem_total_mb": 24564, "utilization_pct": 0, "processes": list(procs)}
    core.SNAPSHOT_PATH.write_text(json.dumps({"ts": time.time() - 5, "mode": "jump", "hosts": {
        "cuda01": {"name": "cuda01", "reachable": True, "error": "", "gpus": [
            gpu(0, 9000, [{"pid": 1, "user": "alice", "mem_used_mb": 9000, "command": "py"}]),
            gpu(1, 1), gpu(2, 1)]},
        "cuda02": {"name": "cuda02", "reachable": False, "error": "timed out", "gpus": []},
    }}))
    text = tui.render_gpu_map(core.read_gpu_snapshot(),
                              [core.get_task(sub.task_id)], ["cuda01", "cuda02"]).plain
    assert "alice" in text and f"#{sub.task_id}" in text
    assert "1/3 free" in text          # GPU 0 is alice's, GPU 1 ours
    assert "unreachable" in text and "network jump" in text

    async def t(app, pilot):
        assert core.SNAPSHOT_REQUEST.exists()      # tui asked the daemon for fresh data
        await pilot.press("g")
        await pilot.pause()
        assert "alice" in str(app.query_one("#gpumap").render())
    run_app(t)


def test_gpu_map_without_snapshot(state):
    assert "No GPU snapshot yet" in tui.render_gpu_map(None, [], []).plain
