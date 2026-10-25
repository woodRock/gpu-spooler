"""The `task` binary without a cluster: config init, the built-in config, help and exit codes."""
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from conftest import task_argv


@pytest.fixture
def home():
    with tempfile.TemporaryDirectory(prefix="tsc.", dir="/tmp") as d:
        yield Path(d)


def task(home: Path, *args: str, **env) -> subprocess.CompletedProcess:
    e = dict(os.environ, HOME=str(home), **env)
    e.pop("GPU_SPOOLER_CONFIG", None)
    e.update(env)
    return subprocess.run(task_argv(*args), env=e, capture_output=True, text=True, timeout=60)


def test_config_init_never_overwrites(home):
    target = home / "hosts.toml"
    r = task(home, "config", "init", GPU_SPOOLER_CONFIG=str(target), LOGNAME="someone", USER="someone")
    assert r.returncode == 0 and target.exists()
    assert "for user 'someone'" in r.stdout and "--user" in r.stdout           # names the user it used
    assert 'mode = "auto"' in target.read_text() and 'user = "someone"' in target.read_text()
    target.write_text("# mine\n")
    r = task(home, "config", "init", GPU_SPOOLER_CONFIG=str(target))
    assert r.returncode != 0 and target.read_text() == "# mine\n"


def test_config_init_user(home):
    target = home / "hosts.toml"
    r = task(home, "config", "init", "-u", "abc1", GPU_SPOOLER_CONFIG=str(target))
    assert r.returncode == 0
    text = target.read_text()
    assert 'user = "abc1"' in text and 'remote_root = "/vol/ecrg-solar/abc1"' in text
    assert task(home, "config", "path", GPU_SPOOLER_CONFIG=str(target)).stdout.strip() == str(target)


def test_list_with_the_builtin_config(home):
    r = task(home, "list", GPU_SPOOLER_CONFIG=str(home / "none.toml"))
    assert r.returncode == 0 and "No tasks." in r.stdout
    r = task(home, "config", "path", GPU_SPOOLER_CONFIG=str(home / "none.toml"))
    assert "does not exist; built-in defaults are used" in r.stdout


def test_help_lists_every_command_and_bad_usage_exits_2(home):
    out = task(home).stdout
    for c in ("submit", "list", "logs", "wait", "kill", "sync", "fetch", "gpus", "exec", "daemon", "config", "tui"):
        assert f"task {c}" in out
    assert task(home, "--version").stdout.startswith("task ")
    assert task(home, "daemon", "restart").returncode == 2
    assert task(home, "kill").returncode == 1                                   # IDs or --all
    assert task(home, "wait", "99").returncode == 1                             # unknown task
