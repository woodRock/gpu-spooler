"""Optional run provenance with taura: the launch script's commit report, the `taura record` calls
(unit, with a fake taura), and the whole flow on the fake cluster with the real taura (integration)."""
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from gpu_spooler import jobs, provenance
from gpu_spooler.config import ConfigError, parse_config

GIT = ["git", "-c", "user.email=t@example.com", "-c", "user.name=t"]


def _row(tmp: Path, **kw) -> dict:
    row = {"id": 12, "command": "python3 train.py --seed 3", "working_dir": str(tmp), "repo_root": str(tmp),
           "server": "cuda01", "gpu_indices": "0,1", "submitted_at": time.mktime((2026, 10, 19, 9, 0, 0, 0, 0, -1)),
           "started_at": 1760839331.0, "finished_at": None, "exit_code": None, "retry_count": 0,
           "git_commit": "a" * 40, "git_dirty": 0}
    return row | kw


def _git_repo(tmp: Path, ledger: bool = True) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(tmp)], check=True)
    if ledger:
        (tmp / "ledger").mkdir()
    return tmp


# ── launch script and config ─────────────────────────────────────────────────

def test_launch_script_reports_the_commit():
    script = jobs.launch_script("12-abc", "~/repo", "0", "true", delay=False)
    assert "TS_GIT" in script and "git rev-parse -q --verify HEAD" in script
    assert jobs.parse_git("TS_PID 4242\nTS_GIT " + "b" * 40 + " 3\n") == ("b" * 40, 3)
    assert jobs.parse_git("TS_PID 4242\n") is None
    assert jobs.parse_launch("TS_GIT " + "b" * 40 + " 0\nTS_PID 7\n") == 7
    assert "TS_DIRTY" in script
    out = "TS_GIT " + "b" * 40 + " 2\nTS_DIRTY results/a.json\nTS_DIRTY figs/x y.png\nTS_PID 7\n"
    assert jobs.parse_dirty(out) == ["results/a.json", "figs/x y.png"]
    assert jobs.parse_dirty("TS_PID 7\n") == []


def test_config_setting():
    assert parse_config({}).taura == "auto"
    assert parse_config({"provenance": {"taura": "off"}}).taura == "off"
    with pytest.raises(ConfigError, match="provenance.taura"):
        parse_config({"provenance": {"taura": "sometimes"}})


def test_environment_probe_runs_like_the_job(tmp_path):
    assert provenance.probe_command("python3 scripts/02.py --device cuda") == "python3"
    assert provenance.probe_command("PYTHONPATH=src /v/.venv/bin/python3.12 s.py") == "PYTHONPATH=src /v/.venv/bin/python3.12"
    assert provenance.probe_command("bash -c 'echo hi'") == "python3"      # not a python command
    script = jobs.launch_script("12-abc", "~/repo", "0", "PYTHONPATH=src python3 x.py", delay=False,
                                probe="PYTHONPATH=src python3")
    assert 'cat > "$D/taura_probe.py"' in script and provenance.PROBE.strip().splitlines()[0] in script
    run_sh = script.split("cat > \"$D/run.sh\"")[1]
    assert run_sh.index("taura_probe.py") < run_sh.index("( PYTHONPATH=src python3 x.py")
    assert "taura_probe" not in jobs.launch_script("12-abc", "~/repo", "0", "true", delay=False)
    # git may only be on a login shell's PATH: the commit report must find it like task sync does
    from gpu_spooler.repo import FIND_GIT
    launch = jobs.launch_script("12-abc", "~/repo", "0", "true", delay=False)
    assert launch.index(FIND_GIT.strip()) < launch.index("TS_GIT")
    # the probe itself, run here: one JSON line
    probe = tmp_path / "probe.py"
    probe.write_text(provenance.PROBE)
    out = subprocess.run(["python3", str(probe)], capture_output=True, text=True, check=True).stdout
    env = __import__("json").loads(out)
    assert env["python"] and isinstance(env["packages"], dict) and isinstance(env["gpus"], list)
    log = tmp_path / "1.log"
    log.write_text("# task 1\n" + provenance.ENV_MARK + out + "training...\n")
    assert provenance.env_from_log(str(log)) == env
    assert provenance.env_from_log(str(tmp_path / "missing.log")) is None


# ── the taura record calls ─────────────────────────────────────────────────────

def test_run_ids_and_record_args(tmp_path):
    repo = _git_repo(tmp_path / "proj")
    (repo / "sub").mkdir()
    row = _row(repo, working_dir=str(repo / "sub"))
    assert provenance.run_id(row) == "2026-10-19-spooler-12"
    assert provenance.run_id(row | {"retry_count": 1}) == "2026-10-19-spooler-12-2"
    start = provenance.record_args("start", row)
    assert start[:6] == ["-C", str(repo), "record", "2026-10-19-spooler-12", "--event", "start"]
    for pair in (["--command", row["command"]], ["--cwd", "sub"], ["--host", "cuda01"], ["--gpus", "0,1"],
                 ["--commit", "a" * 40, "--dirty-count", "0"], ["--spooler-job", "12"], ["--runner", "gpu-spooler"]):
        i = start.index(pair[0])
        assert start[i:i + len(pair)] == pair
    dirty = provenance.record_args("start", row | {"git_dirty": 2, "git_dirty_paths": "results/a.json\nfigs/b.png"})
    i = dirty.index("--dirty-count")
    assert dirty[i:i + 5] == ["--dirty-count", "2", "--dirty-paths", "results/a.json", "figs/b.png"]
    finish = provenance.record_args("finish", row | {"exit_code": 3, "finished_at": 1760839400.0})
    assert finish[finish.index("--exit-code") + 1] == "3"
    out = repo / "results" / "o.txt"
    out.parent.mkdir()
    out.write_text("x")
    fetch = provenance.record_args("fetch", row, outputs=[str(out)])
    assert fetch[fetch.index("--outputs") + 1:] == ["results/o.txt"]
    assert provenance.record_args("fetch", row, outputs=[]) is None
    no_commit = provenance.record_args("start", row | {"git_commit": None})
    assert no_commit[no_commit.index("--commit") + 1] == "unknown"
    # a repo without a taura ledger is left alone
    plain = _git_repo(tmp_path / "plain", ledger=False)
    assert provenance.record_args("start", _row(plain)) is None


def test_notify_uses_taura_only_when_present_and_enabled(tmp_path, monkeypatch):
    repo = _git_repo(tmp_path / "proj")
    bin_dir, calls = tmp_path / "bin", tmp_path / "calls.txt"
    bin_dir.mkdir()
    (bin_dir / "taura").write_text(f'#!/bin/sh\necho "$@" >> {calls}\n')
    (bin_dir / "taura").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    assert provenance.notify("start", _row(repo)) is True
    assert "record 2026-10-19-spooler-12 --event start" in calls.read_text()
    assert provenance.notify("start", _row(repo), setting="off") is False
    assert provenance.notify("start", _row(repo, no_record=1)) is False  # task submit --no-record
    assert calls.read_text().count("\n") == 1
    # at finish, the environment the job printed goes to taura as --env-json
    log = tmp_path / "12.log"
    log.write_text(provenance.ENV_MARK + '{"python": "3.12.15", "packages": {}}\n')
    (bin_dir / "taura").write_text(f'#!/bin/sh\necho "$@" >> {calls}\n'
                                   f'for a in "$@"; do case "$a" in *.json) cat "$a" >> {calls};; esac; done\n')
    assert provenance.notify("finish", _row(repo, exit_code=0, log_file=str(log))) is True
    assert "--env-json" in calls.read_text() and '"python": "3.12.15"' in calls.read_text()
    # a failing taura is logged, never raised
    (bin_dir / "taura").write_text("#!/bin/sh\necho broken >&2\nexit 2\n")
    logged = []
    assert provenance.notify("finish", _row(repo, exit_code=0), log=logged.append) is False
    assert "exited 2" in logged[0]
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert provenance.notify("start", _row(repo)) is False  # taura not installed


# ── the whole flow, with the real taura ──────────────────────────────────────────

@pytest.mark.integration
def test_spooler_tasks_become_taura_runs(ts, fake_cluster):
    taura = shutil.which("taura")
    if taura is None:
        pytest.skip("taura is not installed")
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="tsp.", dir="/tmp")) / f"proj{int(time.time() * 1000)}"
    root.mkdir(parents=True)
    (root / "ledger").mkdir()
    (root / "ledger" / "artifacts.toml").write_text("")
    (root / "train.py").write_text("print('v1')\n")
    (root / "results").mkdir()
    (root / "results" / ".keep").write_text("")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run([*GIT, "-C", str(root), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(root), "commit", "-qm", "init"], check=True)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (root / "train.py").write_text("print('v2, not committed')\n")    # synced as a snapshot of a dirty tree

    ts.run("sync", cwd=str(root))
    tid = ts.submit("submit", "-g", "1", "python3 train.py --seed 5 > results/out.txt", cwd=str(root))
    ts.run("fetch", "results", "--after", str(tid), cwd=str(root), timeout=120)
    assert (root / "results" / "out.txt").read_text() == "v2, not committed\n"

    records = list((root / "ledger" / "runs").glob("*-spooler-*.toml"))
    assert len(records) == 1, records
    rec = subprocess.run([taura, "-C", str(root), "--json", "runs"], capture_output=True, text=True)
    import json
    [run] = json.loads(rec.stdout)["runs"]
    assert run["spooler_job"] == tid and run["runner"] == "gpu-spooler" and run["seed"] == 5
    assert run["status"] == "done" and run["exit_code"] == 0 and run["host"] and run["gpus"] is not None
    assert run["dirty"] is True and run["base_commit"] == head    # the server ran the dirty snapshot
    assert run["commit"] != head
    assert list(run["outputs"]) == ["results/out.txt"]
    env = run["env"]                                                 # reported by the job, not the laptop
    assert env["python"] and env["hash"].startswith("sha256:") and "packages" in env


@pytest.mark.integration
def test_one_fetch_after_two_tasks_records_each_its_own_outputs(ts, fake_cluster):
    taura = shutil.which("taura")
    if taura is None:
        pytest.skip("taura is not installed")
    import json
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="tsp.", dir="/tmp")) / f"proj{int(time.time() * 1000)}"
    (root / "ledger").mkdir(parents=True)
    (root / "ledger" / "artifacts.toml").write_text("")
    (root / "run.sh").write_text('mkdir -p "$1" && sleep 2 && echo "$1" > "$1/summary.csv"\n')
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run([*GIT, "-C", str(root), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(root), "commit", "-qm", "init"], check=True)

    ts.run("sync", cwd=str(root))
    a = ts.submit("submit", "-g", "1", "sh run.sh results/a", cwd=str(root))   # running at once
    b = ts.submit("submit", "-g", "1", "sh run.sh results/b", cwd=str(root))
    out = ts.run("fetch", "results", "--after", str(a), str(b), cwd=str(root), timeout=120).stdout
    assert "Recorded 1 output(s) of task" in out and "could not be told apart" not in out

    rec = subprocess.run([taura, "-C", str(root), "--json", "runs"], capture_output=True, text=True)
    outputs = {r["spooler_job"]: list(r["outputs"]) for r in json.loads(rec.stdout)["runs"]}
    assert outputs == {a: ["results/a/summary.csv"], b: ["results/b/summary.csv"]}
