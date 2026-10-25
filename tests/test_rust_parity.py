"""
The Rust port against the Python version, as black boxes on the fake cluster: the same command,
the same output. Skipped unless GPU_SPOOLER_BIN points at the Rust `task` binary, e.g.

  cargo build --release
  GPU_SPOOLER_BIN=target/release/task .venv/bin/python -m pytest tests/test_rust_parity.py

Each migration step adds the commands it ports.
"""

import os
import re
import subprocess
import sys

import pytest

from conftest import fake

BIN = os.environ.get("GPU_SPOOLER_BIN")
pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not BIN, reason="GPU_SPOOLER_BIN is not set")]

# `task gpus` prints a progress line per host as each answers, in whatever order they do.
PROGRESS = re.compile(r"^  \[ *\d+/\d+\] .*\n", re.M)


def both(ts, *args):
    py = subprocess.run([sys.executable, "-m", "gpu_spooler.task", *args], env=ts.env,
                        capture_output=True, text=True, timeout=120, cwd=str(ts.home))
    rs = subprocess.run([os.path.abspath(BIN), *args], env=ts.env,
                        capture_output=True, text=True, timeout=120, cwd=str(ts.home))
    return py, rs


def progress(out):
    """The progress lines, without the order they came in."""
    return sorted(re.sub(r"\[ *\d+/", "[?/", line) for line in PROGRESS.findall(out))


def same(py, rs):
    assert rs.returncode == py.returncode, (py, rs)
    assert PROGRESS.sub("", rs.stdout) == PROGRESS.sub("", py.stdout)
    assert progress(rs.stdout) == progress(py.stdout)
    assert rs.stderr == py.stderr


@pytest.mark.parametrize("args", [
    ["gpus"], ["gpus", "-d"], ["gpus", "-f"], ["gpus", "-H", "cuda02"], ["gpus", "-d", "-f", "-H", "fake"],
    ["gpus", "--workers", "1"],
])
def test_gpus(ts, args):
    same(*both(ts, *args))


def test_gpus_with_busy_gpus(ts):
    fake("busy", "cuda01", "0")
    fake("busy", "cuda02", "3")
    try:
        for args in (["gpus"], ["gpus", "-d"], ["gpus", "-f", "-d"]):
            same(*both(ts, *args))
    finally:
        fake("idle", "cuda01")
        fake("idle", "cuda02")


def test_gpus_with_a_host_down(ts):
    fake("stop", "cuda02")
    try:
        for args in (["gpus"], ["gpus", "-d"]):
            same(*both(ts, *args))
    finally:
        fake("start", "cuda02")


def test_gpus_errors(ts):
    same(*both(ts, "gpus", "-H", "nope"))


def test_gpus_detective_history_missing(ts):
    cfg = ts.home / "hosts.toml"
    cfg.write_text(open(ts.env["GPU_SPOOLER_CONFIG"]).read()
                   + f'\n[detective]\nhistory = "{ts.home}/none.db"\n')
    ts.env["GPU_SPOOLER_CONFIG"] = str(cfg)
    same(*both(ts, "gpus"))
