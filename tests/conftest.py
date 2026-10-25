import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FAKE = Path(os.environ.get("TS_FAKE_CLUSTER", "/tmp/ts-fake-cluster"))
FAKE_SH = REPO / "dev" / "fake_cluster.sh"

# These tests drive the `task` binary as a black box: GPU_SPOOLER_BIN, or this checkout's release
# build (`cargo build --release`). A `task submit` starts its daemon, so they exercise that too.
BIN = Path(os.environ.get("GPU_SPOOLER_BIN") or REPO / "target" / "release" / "task").resolve()


def task_argv(*args: str) -> list[str]:
    """How to run `task ARGS` in the tests."""
    if not BIN.exists():
        pytest.fail(f"{BIN} doesn't exist: run `cargo build --release` (or set GPU_SPOOLER_BIN)")
    return [str(BIN), *args]


def _sshd() -> str | None:
    return shutil.which("sshd") or (os.path.exists("/usr/sbin/sshd") and "/usr/sbin/sshd") or None


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: needs a local sshd (fake cluster)")


@pytest.fixture(scope="session")
def fake_cluster():
    if not _sshd():
        pytest.skip("sshd not available")
    status = subprocess.run([str(FAKE_SH), "status"], capture_output=True, text=True).stdout
    already_up = (FAKE / "hosts.toml").exists() and "down" not in status
    if not already_up:
        r = subprocess.run([str(FAKE_SH), "up"], capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip(f"fake cluster failed to start: {r.stderr or r.stdout}")
    yield FAKE
    if not already_up:
        subprocess.run([str(FAKE_SH), "down"], capture_output=True)


def fake(*args: str) -> str:
    return subprocess.run([str(FAKE_SH), *args], check=True, capture_output=True,
                          text=True).stdout


class TS:
    """An isolated gpu-spooler environment (own HOME) driven via the CLI."""

    def __init__(self, home: Path, config: Path):
        self.home = home
        self.state = home / ".gpu-spooler"
        self.env = dict(os.environ, HOME=str(home), GPU_SPOOLER_CONFIG=str(config),
                        GPU_SPOOLER_STATUS_INTERVAL="1", NO_COLOR="1")

    def run(self, *args: str, check: bool = True, timeout: float = 120,
            cwd: str | None = None) -> subprocess.CompletedProcess:
        r = subprocess.run(task_argv(*args),
                           env=self.env, capture_output=True, text=True,
                           timeout=timeout, cwd=cwd or str(self.home))
        if check and r.returncode != 0:
            raise AssertionError(f"task {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
        return r

    def db(self):
        import sqlite3
        con = sqlite3.connect(str(self.state / "tasks.db"))
        con.row_factory = sqlite3.Row
        return con

    def row(self, task_id: int):
        with self.db() as con:
            return dict(con.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone())

    def wait_status(self, task_id: int, statuses, timeout: float = 60) -> dict:
        statuses = {statuses} if isinstance(statuses, str) else set(statuses)
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = self.row(task_id)
            if r["status"] in statuses:
                return r
            time.sleep(0.5)
        log = (self.state / "daemon.log")
        raise AssertionError(
            f"task {task_id} still {self.row(task_id)['status']} after {timeout}s\n"
            + (log.read_text()[-3000:] if log.exists() else ""))

    def submit(self, *args: str, cwd: str | None = None) -> int:
        out = self.run(*args, cwd=cwd).stdout
        for line in out.splitlines():
            if line.startswith("Queued task "):
                return int(line.split()[2].rstrip(":"))
        raise AssertionError(out)


@pytest.fixture
def ts(fake_cluster):
    # Short path: macOS limits unix socket paths (ControlPath) to ~104 bytes.
    home = Path(tempfile.mkdtemp(prefix="tsh.", dir="/tmp"))
    env = TS(home, fake_cluster / "hosts.toml")
    yield env
    env.run("daemon", "stop", check=False)
    shutil.rmtree(home, ignore_errors=True)
