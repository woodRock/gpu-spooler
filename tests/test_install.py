"""
install.sh, run the way the one-liner runs it (piped into bash), against a fake release folder
built from the local Rust binary: GPU_SPOOLER_RELEASES points at it, and HOME and the bin
directory are temporary. Skipped unless GPU_SPOOLER_BIN points at the Rust `task` binary.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BIN = os.environ.get("GPU_SPOOLER_BIN")
pytestmark = pytest.mark.skipif(not BIN, reason="GPU_SPOOLER_BIN is not set")

TARGET = {("Darwin", "arm64"): "aarch64-apple-darwin", ("Darwin", "x86_64"): "x86_64-apple-darwin",
          ("Linux", "x86_64"): "x86_64-unknown-linux-musl", ("Linux", "aarch64"): "aarch64-unknown-linux-musl"}[
    (os.uname().sysname, os.uname().machine)]


def release(folder: Path, binary: Path, good_sha: bool = True) -> None:
    """A release folder like GitHub's latest/download: the archive and its .sha256."""
    folder.mkdir(parents=True, exist_ok=True)
    name = f"gpu-spooler-{TARGET}.tar.gz"
    with tarfile.open(folder / name, "w:gz") as tf:
        tf.add(binary, arcname="task")
    digest = hashlib.sha256((folder / name).read_bytes()).hexdigest() if good_sha else "0" * 64
    (folder / f"{name}.sha256").write_text(f"{digest}  {name}\n")


class Machine:
    """A temporary home with its own bin directory and a fake release to install from."""

    def __init__(self, root: Path):
        self.home = root / "home"
        self.bin = self.home / ".local" / "bin"
        self.releases = root / "releases"
        self.home.mkdir()
        release(self.releases, Path(BIN).resolve())
        self.env = dict(os.environ, HOME=str(self.home), GPU_SPOOLER_BIN_DIR=str(self.bin),
                        GPU_SPOOLER_RELEASES=f"file://{self.releases}",
                        PATH=f"{self.bin}:/usr/bin:/bin:/usr/sbin:/sbin")
        self.env.pop("GPU_SPOOLER_CONFIG", None)

    def install(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        r = subprocess.run(["bash", "-s", "--", *args], input=(REPO / "install.sh").read_text(), env=self.env,
                           capture_output=True, text=True, timeout=120, cwd=str(self.home))
        if check and r.returncode != 0:
            raise AssertionError(f"install.sh {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
        return r

    def task(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.bin / "task"), *args], env=self.env, capture_output=True, text=True,
                              timeout=60)


@pytest.fixture
def machine():
    root = Path(tempfile.mkdtemp(prefix="tsi.", dir="/tmp"))
    m = Machine(root)
    yield m
    if (m.bin / "task").exists():
        m.task("daemon", "stop")
    shutil.rmtree(root, ignore_errors=True)


def version() -> str:
    return subprocess.run([BIN, "--version"], capture_output=True, text=True).stdout.strip()


def test_fresh_install_upgrade_and_uninstall(machine):
    r = machine.install()
    assert (machine.bin / "task").is_file() and not (machine.bin / "task").is_symlink()
    assert machine.task("--version").stdout.strip() == version()
    assert "SHA-256 checked" in r.stdout and "Done. Try: task --help" in r.stdout
    assert machine.task("list").stdout.strip() == "No tasks."
    r = machine.install("--upgrade")
    assert "Upgraded gpu-spooler" in r.stdout
    (machine.home / ".gpu-spooler" / "hosts.toml").write_text("# mine\n")
    r = machine.install("--uninstall")
    assert not (machine.bin / "task").exists()
    assert (machine.home / ".gpu-spooler" / "hosts.toml").read_text() == "# mine\n"     # state kept
    assert "kept in" in r.stdout


def test_a_bad_checksum_installs_nothing(machine):
    release(machine.releases, Path(BIN).resolve(), good_sha=False)
    r = machine.install(check=False)
    assert r.returncode != 0 and "SHA-256 mismatch" in r.stderr
    assert not (machine.bin / "task").exists()


def test_someone_elses_task_is_left_alone_unless_forced(machine):
    machine.bin.mkdir(parents=True)
    other = machine.bin / "task"
    other.write_text("#!/bin/sh\necho 2.6.2\n")             # Taskwarrior's `task --version`
    other.chmod(0o755)
    r = machine.install(check=False)
    assert r.returncode == 1 and "another program's `task`" in r.stdout
    assert other.read_text().endswith("echo 2.6.2\n")
    r = machine.install("--uninstall")
    assert other.exists() and "isn't gpu-spooler's; left alone" in r.stderr
    machine.install("--force")
    assert machine.task("--version").stdout.strip() == version()


def test_dry_run_changes_nothing(machine):
    r = machine.install("--dry-run")
    assert "+ curl" in r.stdout and not machine.bin.exists()


def python_dev_install(machine) -> None:
    """What dev/install-dev.sh leaves: links into a checkout's .venv (this one's)."""
    machine.bin.mkdir(parents=True, exist_ok=True)
    for c in ("task", "gpu-status", "tui"):
        os.symlink(REPO / ".venv" / "bin" / c, machine.bin / c)


@pytest.mark.skipif(not (REPO / ".venv" / "bin" / "task").exists(), reason="no Python dev install in this checkout")
def test_replaces_the_python_dev_install(machine):
    python_dev_install(machine)
    r = machine.install()
    assert not (machine.bin / "task").is_symlink() and machine.task("--version").stdout.strip() == version()
    assert not os.path.lexists(machine.bin / "gpu-status") and not os.path.lexists(machine.bin / "tui")
    assert "now `task gpus`" in r.stdout and "now `task tui`" in r.stdout


@pytest.mark.integration
@pytest.mark.skipif(not (REPO / ".venv" / "bin" / "task").exists(), reason="no Python dev install in this checkout")
def test_switch_over_keeps_a_running_job(machine, fake_cluster):
    """The switch-over: a job runs under the Python daemon; install.sh replaces the Python version
    with the Rust one, which adopts the job and finishes it."""
    python_dev_install(machine)
    env = dict(machine.env, GPU_SPOOLER_CONFIG=str(fake_cluster / "hosts.toml"), GPU_SPOOLER_STATUS_INTERVAL="1",
               PATH=f"{machine.bin}:{os.environ['PATH']}")
    machine.env = env
    r = subprocess.run([str(machine.bin / "task"), "submit", "-q", "-g", "0", "-H", "local", "-C", "/tmp", "--",
                        "for i in 1 2 3 4 5 6 7 8; do echo line$i; sleep 1; done"], env=env, capture_output=True,
                       text=True, timeout=60)
    tid = int(r.stdout.strip())
    state = machine.home / ".gpu-spooler"
    py_pid = int((state / "daemon.pid").read_text())
    assert "gpu_spooler.daemon" in subprocess.run(["ps", "-o", "command=", "-p", str(py_pid)],
                                                  capture_output=True, text=True).stdout
    import sqlite3
    status = lambda: sqlite3.connect(state / "tasks.db").execute(
        "SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()[0]
    deadline = time.time() + 30
    while status() != "running":
        assert time.time() < deadline
        time.sleep(0.3)
    time.sleep(2)

    out = machine.install().stdout
    assert "stopping the daemon" in out and "daemon restarted with the new version" in out
    rs_pid = int((state / "daemon.pid").read_text())
    assert rs_pid != py_pid and "__daemon" in subprocess.run(["ps", "-o", "command=", "-p", str(rs_pid)],
                                                             capture_output=True, text=True).stdout
    deadline = time.time() + 60
    while status() not in ("done", "failed", "lost"):
        assert time.time() < deadline, (state / "daemon.log").read_text()[-2000:]
        time.sleep(0.5)
    assert status() == "done"
    body = (state / "logs" / f"{tid}.log").read_text().split("─" * 60 + "\n", 1)[1]
    assert body == "".join(f"line{i}\n" for i in range(1, 9))         # every line, once
    assert "Re-adopting 1 running task(s)." in (state / "daemon.log").read_text()
