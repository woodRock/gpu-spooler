"""
install.sh, run the way the one-liner runs it (piped into bash), against a fake release folder
built from the local Rust binary: GPU_SPOOLER_RELEASES points at it, and HOME and the bin
directory are temporary. (The switch-over from a running Python daemon, which needs the Python
version, was tested before it was removed: see the commit that added this file.)
"""

import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
from conftest import BIN

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
        release(self.releases, BIN)
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
    return subprocess.run([str(BIN), "--version"], capture_output=True, text=True).stdout.strip()


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
    release(machine.releases, BIN, good_sha=False)
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
    """What the Python version's dev/install-dev.sh left: links into a checkout's .venv, to entry
    points that import gpu_spooler (which no longer runs once the checkout has moved on)."""
    venv = machine.home / "old-checkout" / ".venv" / "bin"
    venv.mkdir(parents=True)
    machine.bin.mkdir(parents=True, exist_ok=True)
    for c, mod in (("task", "task"), ("gpu-status", "gpu_status"), ("tui", "tui")):
        (venv / c).write_text(f"#!/nonexistent/python3\nfrom gpu_spooler.{mod} import main\nmain()\n")
        (venv / c).chmod(0o755)
        os.symlink(venv / c, machine.bin / c)


def test_replaces_the_python_dev_install(machine):
    python_dev_install(machine)
    r = machine.install()
    assert not (machine.bin / "task").is_symlink() and machine.task("--version").stdout.strip() == version()
    assert not os.path.lexists(machine.bin / "gpu-status") and not os.path.lexists(machine.bin / "tui")
    assert "now `task gpus`" in r.stdout and "now `task tui`" in r.stdout
