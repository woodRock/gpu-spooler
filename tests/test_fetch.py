"""task fetch: the whole workflow against the fake cluster (the rules themselves are unit-tested in
crates/gpu-spooler/src/fetch.rs)."""
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

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


# ── resolving ────────────────────────────────────────────────────────────────


# ── planning: what is replaced and what is kept ──────────────────────────────


# ── against the fake cluster ─────────────────────────────────────────────────

# ── which task wrote what ────────────────────────────────────────────────────


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
