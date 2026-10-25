"""Run provenance with taura: the whole flow on the fake cluster with the real taura (the `taura
record` arguments are unit-tested in crates/gpu-spooler/src/provenance.rs)."""
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

GIT = ["git", "-c", "user.email=t@example.com", "-c", "user.name=t"]


# ── launch script and config ─────────────────────────────────────────────────


# ── the taura record calls ─────────────────────────────────────────────────────


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
