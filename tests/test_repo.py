"""Remote git copies: --repo-status, --sync, and submit warnings (fake cluster)."""
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from gpu_spooler.repo import sh_path

GIT = ["git", "-c", "user.email=t@example.com", "-c", "user.name=t"]


def test_sh_path():
    assert sh_path("/a b/c") == "'/a b/c'"
    assert sh_path("~/x y") == "\"$HOME\"/'x y'"
    assert sh_path("~") == '"$HOME"'


@pytest.fixture
def project():
    root = Path(tempfile.mkdtemp(prefix="tsr.", dir="/tmp")) / f"proj{int(time.time() * 1000)}"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "run.py").write_text('print("version 1")\n')
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run([*GIT, "-C", str(root), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(root), "commit", "-qm", "init"], check=True)
    return root


@pytest.mark.integration
def test_sync_workflow(ts, project, fake_cluster):
    sub = str(project / "sub")
    remote = fake_cluster / "repos" / project.name

    r = ts.run("sync", "--check", cwd=sub, check=False)
    assert r.returncode == 1 and "doesn't exist" in r.stdout
    out = ts.run("submit", "-g", "1", "-H", "cuda01", "true", cwd=sub).stdout
    ts.run("kill", "1")
    assert "doesn't exist" in out and "task sync`" in out

    out = ts.run("sync", cwd=sub).stdout
    assert "OK" in out
    assert (remote / "sub" / "run.py").read_text() == 'print("version 1")\n'
    assert "up to date" in ts.run("sync", "--check", cwd=sub).stdout

    # Uncommitted local edits are stale until synced, then included.
    (project / "sub" / "run.py").write_text('print("version 2")\n')
    assert "run `task sync`" in ts.run("sync", "--check", cwd=sub, check=False).stdout
    ts.run("sync", cwd=sub)
    assert (remote / "sub" / "run.py").read_text() == 'print("version 2")\n'
    status = subprocess.run(["git", "-C", str(project), "status", "--porcelain"],
                            capture_output=True, text=True).stdout
    assert "run.py" in status                      # local working tree untouched

    tid = ts.submit("submit", "-g", "1", "python3 run.py", cwd=sub)
    r = ts.wait_status(tid, ("done", "failed", "lost"), 40)
    assert r["status"] == "done" and r["remote_dir"] == f"{remote}/sub"
    assert "version 2" in (ts.state / "logs" / f"{tid}.log").read_text()


@pytest.mark.integration
def test_sync_refuses_to_clobber_remote_edits(ts, project, fake_cluster):
    sub = str(project / "sub")
    ts.run("sync", cwd=sub)
    remote = fake_cluster / "repos" / project.name
    (remote / "sub" / "run.py").write_text('print("edited on server")\n')
    assert "uncommitted change" in ts.run("sync", "--check", cwd=sub, check=False).stdout
    (project / "sub" / "run.py").write_text('print("local")\n')
    r = ts.run("sync", cwd=sub, check=False)
    assert r.returncode == 1 and "checkout failed" in r.stdout
    assert (remote / "sub" / "run.py").read_text() == 'print("edited on server")\n'


@pytest.mark.integration
def test_sync_outside_repo_errors(ts):
    r = ts.run("sync", cwd="/tmp", check=False)
    assert r.returncode != 0 and "not inside a git repo" in r.stderr
