"""task fetch: path resolution and the keep/replace rules (unit), and the whole
workflow against the fake cluster (integration)."""
import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from gpu_spooler import fetch
from gpu_spooler.config import parse_config

GIT = ["git", "-c", "user.email=t@example.com", "-c", "user.name=t"]


def _repo(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="tsf.", dir="/tmp")) / f"proj{int(time.time() * 1000)}"
    root.mkdir(parents=True)
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run([*GIT, "-C", str(root), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(root), "commit", "-qm", "init"], check=True)
    return root


def _cfg(two_groups=False, path_map=None):
    data = {"groups": {"g": {"hosts": ["a", "b"], "remote_root": "/vol/me"}}}
    if path_map:
        data["groups"]["g"]["path_map"] = path_map
    if two_groups:
        data["groups"]["h"] = {"hosts": ["c"], "remote_root": "/vol/other"}
    return parse_config(data)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── resolving ────────────────────────────────────────────────────────────────

def test_resolve_maps_repo_paths_like_jobs():
    root = _repo({"results/x.txt": "x"})
    cfg = _cfg()
    group, items = fetch.resolve(cfg, cfg.select("g"), ["results", "results/x.txt"], str(root))
    assert [h.name for h in group] == ["a", "b"]
    assert [i.remote for i in items] == [f"/vol/me/{root.name}/results",
                                         f"/vol/me/{root.name}/results/x.txt"]
    assert items[0].local == str(root / "results")
    _, items = fetch.resolve(cfg, cfg.select("g"), ["x.txt"], str(root / "results"))
    assert items[0].remote == f"/vol/me/{root.name}/results/x.txt"   # relative to the cwd


def test_resolve_refuses_paths_outside_the_repo_and_ambiguous_groups():
    root = _repo({"a": "a"})
    cfg = _cfg()
    with pytest.raises(fetch.FetchError, match="outside the repo"):
        fetch.resolve(cfg, cfg.select("g"), ["../elsewhere"], str(root))
    with pytest.raises(fetch.FetchError, match="REMOTE:LOCAL"):
        fetch.resolve(cfg, cfg.select("g"), ["x"], "/tmp")             # not in a repo
    cfg2 = _cfg(two_groups=True)
    with pytest.raises(fetch.FetchError, match="-H"):
        fetch.resolve(cfg2, cfg2.select(None), ["a"], str(root))
    _, items = fetch.resolve(cfg2, cfg2.select("h"), ["a"], str(root))
    assert items[0].remote == f"/vol/other/{root.name}/a"


def test_resolve_explicit_and_path_map():
    root = _repo({"a": "a"})
    cfg = _cfg()
    _, items = fetch.resolve(cfg, cfg.select("g"), ["/vol/run/results:out", "~/r:"], str(root))
    assert (items[0].remote, items[0].local) == ("/vol/run/results", str(root / "out"))
    assert (items[1].remote, items[1].local) == ("~/r", str(root))
    cfg = _cfg(path_map=[{"local": str(root), "remote": "/vol/mapped"}])
    _, items = fetch.resolve(cfg, cfg.select("g"), ["sub/f"], str(root))
    assert items[0].remote == "/vol/mapped/sub/f"                     # path_map wins


# ── planning: what is replaced and what is kept ──────────────────────────────

def test_plan_replaces_only_what_can_be_recovered(tmp_path, monkeypatch):
    root = _repo({"res/tracked_clean": "old", "res/tracked_dirty": "old"})
    res = root / "res"
    (res / "tracked_dirty").write_text("my edit")
    (res / "untracked").write_text("mine")
    (res / "same").write_text("same")
    (res / "fetched_before").write_text("v1")
    (res / "is_a_dir").mkdir()
    monkeypatch.setattr(fetch, "MANIFEST", tmp_path / "fetched.json")
    manifest = {str(res / "fetched_before"): fetch._stamp(str(res / "fetched_before"))}
    desc = {"kind": "dir", "links": {"link": "same"},
            "files": {n: _sha("remote") for n in (
                "tracked_clean", "tracked_dirty", "untracked", "fetched_before", "is_a_dir", "new")}
            | {"same": _sha("same")}}
    item = fetch.Item("res", str(res), "/remote/res")
    plan = fetch.plan_item(item, desc, manifest, force=False)
    assert sorted(plan.new) == ["link", "new"]
    assert sorted(plan.update) == ["fetched_before", "tracked_clean"]  # manifest / git has it
    assert sorted(plan.kept) == ["tracked_dirty", "untracked"]
    assert plan.same == ["same"] and plan.conflict == ["is_a_dir"]
    forced = fetch.plan_item(item, desc, manifest, force=True)
    assert sorted(forced.update) == ["fetched_before", "tracked_clean", "tracked_dirty",
                                     "untracked"] and forced.conflict == ["is_a_dir"]


def test_plan_file_into_an_existing_directory(tmp_path):
    item = fetch.Item("/r/x.txt:.", str(tmp_path), "/r/x.txt")
    plan = fetch.plan_item(item, {"kind": "file", "sha": _sha("x")}, {}, False)
    assert plan.new == [""] and item.local == str(tmp_path / "x.txt")


# ── against the fake cluster ─────────────────────────────────────────────────

# ── which task wrote what ────────────────────────────────────────────────────

def _task(tid, command, wdir, started=1000.0, finished=2000.0):
    return {"id": tid, "command": command, "working_dir": str(wdir),
            "started_at": started, "finished_at": finished}


def _written(root: Path, rel: str, mtime: float = 1500.0) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(rel)
    os.utime(p, (mtime, mtime))
    return str(p)


def test_attribute_by_the_paths_each_command_names(tmp_path):
    # tasks 85 and 86 of the red-herring rerun: same time, same script, different --out
    a = _task(85, "PYTHONPATH=src python3 -m red_herring.experiment --out results/ungrouped", tmp_path)
    b = _task(86, "PYTHONPATH=src python3 -m red_herring.experiment --groups tissue "
                  "--out=results/grouped", tmp_path)
    ua, ga = _written(tmp_path, "results/ungrouped/summary.csv"), _written(tmp_path, "results/grouped/s.csv")
    by_task, shared = fetch.attribute([a, b], [ua, ga])
    assert by_task == {85: [ua], 86: [ga]} and shared == []


def test_attribute_prefers_the_most_specific_name(tmp_path):
    a = _task(1, "python3 sweep.py --out results", tmp_path)
    b = _task(2, "python3 eval.py --out results/eval", tmp_path)
    top, ev = _written(tmp_path, "results/sweep.csv"), _written(tmp_path, "results/eval/e.csv")
    by_task, shared = fetch.attribute([a, b], [top, ev])
    assert by_task == {1: [top], 2: [ev]} and shared == []


def test_attribute_falls_back_to_when_each_task_ran(tmp_path):
    a = _task(1, "python3 train.py", tmp_path, started=1000, finished=2000)
    b = _task(2, "python3 train.py", tmp_path, started=5000, finished=6000)
    early = _written(tmp_path, "results/early.txt", mtime=2050)    # within the slack
    late = _written(tmp_path, "results/late.txt", mtime=5500)
    by_task, shared = fetch.attribute([a, b], [early, late])
    assert by_task == {1: [early], 2: [late]} and shared == []


def test_attribute_credits_every_task_when_nothing_tells_them_apart(tmp_path):
    a, b = _task(1, "python3 train.py", tmp_path), _task(2, "python3 train.py", tmp_path)
    p = _written(tmp_path, "results/out.txt")
    by_task, shared = fetch.attribute([a, b], [p])
    assert by_task == {1: [p], 2: [p]} and shared == [p]


def test_attribute_one_task_gets_everything(tmp_path):
    # `.` and the working dir name every file, so they don't count as naming one
    a = _task(7, "cd . && python3 train.py --data /", tmp_path, started=None, finished=None)
    p = _written(tmp_path, "results/out.txt", mtime=9e9)
    assert fetch.attribute([a], [p]) == ({7: [p]}, [])
    named = fetch._named_paths(a)
    assert str(tmp_path / "train.py") in named
    assert str(tmp_path) not in named and "/" not in named
    assert fetch._named_paths(_task(8, "python3 'unbalanced", tmp_path))   # not shell-parsable


@pytest.mark.integration
def test_fetch_workflow(ts, fake_cluster):
    root = _repo({"sub/run.py": "print(1)\n", "results/r.txt": "committed\n"})
    sub, results = str(root / "sub"), root / "results"
    ts.run("sync", cwd=sub)
    remote = fake_cluster / "repos" / root.name
    (remote / "results" / "r.txt").write_text("from the server\n")
    (remote / "results" / "deep").mkdir()
    (remote / "results" / "deep" / "n.txt").write_text("new\n")
    os.symlink("r.txt", remote / "results" / "inside_link")
    os.symlink("../../..", remote / "results" / "outside_link")
    os.symlink(str(fake_cluster), remote / "results" / "absolute_link")

    out = ts.run("fetch", "results", "--dry-run", cwd=str(root)).stdout
    assert "Would fetch" in out and not (results / "deep").exists()

    out = ts.run("fetch", "results", cwd=str(root)).stdout
    assert (results / "r.txt").read_text() == "from the server\n"     # tracked, clean
    assert (results / "deep" / "n.txt").read_text() == "new\n"
    assert os.readlink(results / "inside_link") == "r.txt"            # a link, not a copy
    assert not os.path.lexists(results / "outside_link")
    assert not os.path.lexists(results / "absolute_link")
    assert "2 skipped (unsafe link)" in out
    assert not list(results.glob(".task-fetch-*"))                    # temp dir removed

    # A fetched file edited locally is kept, then replaced with --force.
    (results / "deep" / "n.txt").write_text("my edit\n")
    (remote / "results" / "deep" / "n.txt").write_text("newer\n")
    out = ts.run("fetch", "results", cwd=sub + "/..").stdout
    assert (results / "deep" / "n.txt").read_text() == "my edit\n" and "kept" in out
    ts.run("fetch", "results", "--force", cwd=str(root))
    assert (results / "deep" / "n.txt").read_text() == "newer\n"

    # Explicit REMOTE:LOCAL, a single file into a directory.
    dest = root / "elsewhere"
    dest.mkdir()
    ts.run("fetch", f"{remote}/results/r.txt:elsewhere", cwd=str(root))
    assert (dest / "r.txt").read_text() == "from the server\n"

    r = ts.run("fetch", "results/../..", cwd=str(root), check=False)
    assert r.returncode != 0 and "outside the repo" in r.stderr


@pytest.mark.integration
def test_fetch_after_waits_for_the_task(ts, fake_cluster):
    root = _repo({"results/.keep": ""})
    ts.run("sync", cwd=str(root))
    tid = ts.submit("submit", "-g", "1", "sleep 3; echo done > results/out.txt", cwd=str(root))
    ts.run("fetch", "results", "--after", str(tid), cwd=str(root), timeout=90)
    assert (root / "results" / "out.txt").read_text() == "done\n"
    bad = ts.submit("submit", "-g", "1", "exit 4", cwd=str(root))
    r = ts.run("fetch", "results", "--after", str(bad), cwd=str(root), check=False,
               timeout=120)
    assert r.returncode != 0 and "did not finish cleanly" in r.stderr
