"""
The Rust port against the Python version, as black boxes on the fake cluster: the same command,
the same output. Skipped unless GPU_SPOOLER_BIN points at the Rust `task` binary, e.g.

  cargo build --release
  GPU_SPOOLER_BIN=target/release/task .venv/bin/python -m pytest tests/test_rust_parity.py

Each migration step adds the commands it ports.
"""

import os
import re
import signal
import subprocess
import sys
import time

import pytest

from conftest import fake

BIN = os.environ.get("GPU_SPOOLER_BIN")
pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not BIN, reason="GPU_SPOOLER_BIN is not set")]

# `task gpus` prints a progress line per host as each answers, in whatever order they do.
PROGRESS = re.compile(r"^  \[ *\d+/\d+\] .*\n", re.M)


PY = [sys.executable, "-m", "gpu_spooler.task"]


def both(ts, *args, input=None):
    """Run `task ARGS` with the Python version, then with the Rust one."""
    def one(argv):
        return subprocess.run([*argv, *args], env=ts.env, capture_output=True, text=True, timeout=120,
                              cwd=str(ts.home), input=input)
    return one(PY), one([os.path.abspath(BIN)])


def progress(out):
    """The progress lines, without the order they came in."""
    return sorted(re.sub(r"\[ *\d+/", "[?/", line) for line in PROGRESS.findall(out))


def same(py, rs):
    what = " ".join(rs.args[1:])
    assert rs.returncode == py.returncode, (what, py, rs)
    assert PROGRESS.sub("", rs.stdout) == PROGRESS.sub("", py.stdout), what
    assert progress(rs.stdout) == progress(py.stdout), what
    assert rs.stderr == py.stderr, what


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


# ── step 2: the queue (read only) and the cluster ─────────────────────────────

def test_list_without_tasks(ts):
    same(*both(ts, "list"))


def test_queue_commands(ts):
    """A queue the Python version built, read by both."""
    done = ts.submit("submit", "--", "echo hello; echo to-stderr >&2")
    failed = ts.submit("submit", "-n", "labelled  run\nwith newline", "--", "echo out; exit 3")
    cpu = ts.submit("submit", "-g", "0", "-H", "local", "--", "echo cpu")
    queued = ts.submit("submit", "-g", "4", "-H", "cuda01", "-m", "20", "-t", "RTX A5000", "--", "true")
    cancelled = ts.submit("submit", "-g", "4", "-H", "cuda01", "--", "true")
    ts.run("kill", str(cancelled))
    for tid, status in ((done, "done"), (failed, "failed"), (cpu, "done")):
        ts.wait_status(tid, status, timeout=90)

    for args in (["list"], ["ls", "-a"], ["list", "--al"], ["info", "99"], ["logs", "99"],
                 ["logs", str(queued)], ["logs", str(cancelled)], ["logs", "-f", str(cancelled)],
                 ["wait", "99"], ["wait", str(done)], ["wait", str(failed)], ["wait", str(cancelled)],
                 ["wait", str(cpu), str(done), str(failed), str(done)],
                 ["wait", str(queued), "--timeout", "0.5"], ["wait", str(done), str(queued), "--timeout", "1"]):
        same(*both(ts, *args))
    for tid in (done, failed, cpu, queued, cancelled):
        same(*both(ts, "info", str(tid)))
        if tid not in (queued, cancelled):
            same(*both(ts, "logs", str(tid)))


def interrupted(ts, *args, after=1.5):
    """Run `task ARGS` with both versions and press Ctrl+C (SIGINT to the process group)."""
    def one(argv):
        p = subprocess.Popen([*argv, *args], env=ts.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, cwd=str(ts.home), start_new_session=True)
        time.sleep(after)
        os.killpg(p.pid, signal.SIGINT)
        out, err = p.communicate(timeout=30)
        return subprocess.CompletedProcess(p.args, p.returncode, out, err)
    return one(PY), one([os.path.abspath(BIN)])


def test_ctrl_c_stops_following_and_waiting(ts):
    done = ts.submit("submit", "--", "echo line1; echo line2")
    queued = ts.submit("submit", "-g", "4", "-H", "cuda01", "--", "true")
    ts.wait_status(done, "done")
    same(*interrupted(ts, "logs", "-f", str(done)))
    same(*interrupted(ts, "wait", str(queued)))


def test_hosts(ts):
    same(*both(ts, "hosts"))
    fake("stop", "cuda02")
    try:
        same(*both(ts, "hosts"))
    finally:
        fake("start", "cuda02")


def test_exec(ts):
    for args in (["exec", "--", "echo hi-$FAKE_HOST"], ["exec", "-H", "cuda01", "--", "echo bad; exit 3"],
                 ["exec", "-H", "cuda02", "true"], ["exec", "-H", "local", "--", "printf ''; false"],
                 ["exec"], ["exec", "-H", "nope", "--", "true"]):
        same(*both(ts, *args))


def test_ssh(ts):
    same(*both(ts, "ssh", "nope"))
    same(*both(ts, "ssh", "cuda01", input="echo hi-$FAKE_HOST; exit 4\n"))


def test_connect(ts):
    same(*both(ts, "connect"))


def test_daemon_status(ts):
    same(*both(ts, "daemon", "status"))
    ts.submit("submit", "-g", "0", "-H", "local", "--", "true")       # starts the daemon
    same(*both(ts, "daemon", "status"))


def test_config_init_and_path(ts):
    target = ts.home / "new" / "hosts.toml"
    ts.env["GPU_SPOOLER_CONFIG"] = str(target)
    same(*both(ts, "config", "path"))
    for args in (["config", "init", "-u", "abc1"], ["config", "init"], ["config", "init", "--user", "x y"]):
        py = subprocess.run([*PY, *args], env=ts.env, capture_output=True, text=True)
        written = target.read_text()
        target.unlink()
        rs = subprocess.run([os.path.abspath(BIN), *args], env=ts.env, capture_output=True, text=True)
        same(py, rs)
        assert target.read_text() == written, args
        same(*both(ts, "config", "path"))
        same(*both(ts, *args))                                          # already exists
        target.unlink()


# ── step 3: commands that change the queue ────────────────────────────────────

def twins(ts):
    """Two fresh homes on the same cluster: (python's, rust's), so task IDs line up."""
    import shutil
    import tempfile
    from pathlib import Path
    from conftest import TS
    homes = [TS(Path(tempfile.mkdtemp(prefix="tsh.", dir="/tmp")), Path(ts.env["GPU_SPOOLER_CONFIG"]))
             for _ in range(2)]
    yield homes
    for h in homes:
        subprocess.run([*PY, "daemon", "stop"], env=h.env, capture_output=True)
        subprocess.run([os.path.abspath(BIN), "daemon", "stop"], env=h.env, capture_output=True)
        shutil.rmtree(h.home, ignore_errors=True)


@pytest.fixture
def pair(ts):
    yield from twins(ts)


def each(pair, *args, input=None):
    """`task ARGS` with Python in one home and Rust in the other, from the same directory."""
    py_home, rs_home = pair
    def one(home, argv):
        r = subprocess.run([*argv, *args], env=home.env, capture_output=True, text=True, timeout=120,
                           cwd="/tmp", input=input)
        # The homes differ only in their names (and the daemon's PID).
        clean = lambda s: re.sub(r"PID \d+", "PID N", s.replace(str(home.home), "{HOME}"))
        return subprocess.CompletedProcess(r.args, r.returncode, clean(r.stdout), clean(r.stderr))
    return one(py_home, PY), one(rs_home, [os.path.abspath(BIN)])


def test_submit_messages(pair):
    for args in (["submit", "-g", "0", "-H", "local", "-n", "first", "--", "echo", "it's", "a b"],
                 ["submit", "-g", "2", "-m", "20", "-H", "cuda01,cuda02", "-t", " RTX A5000, ,A40 ", "--", "true"],
                 ["submit", "-q", "-g", "0", "-H", "cuda01", "-C", "/tmp", "--", "echo cpu nowhere"],
                 ["submit", "-g", "1", "-H", "fake", "python3", "-c", "print(1)", "--flag"],
                 ["submit"], ["submit", "--", "  "], ["submit", "-g", "-1", "true"],
                 ["submit", "-g", "0", "-t", "A40", "true"], ["submit", "-H", "nope", "true"],
                 ["resubmit", "1"], ["resubmit", "-q", "2"], ["resubmit", "99"]):
        same(*each(pair, *args))
    for home in pair:
        home.wait_status(1, "done", 60)
        home.wait_status(2, "done", 60)
    same(*each(pair, "list"))


def test_kill_and_clear_messages(pair):
    for home in pair:
        home.submit("submit", "-g", "0", "-H", "local", "--", "echo done")      # 1
    for home in pair:
        home.wait_status(1, "done", 60)
    for args in (["submit", "-q", "-g", "4", "-H", "cuda01", "--", "true"],          # 2: never fits
                 ["submit", "-q", "-g", "4", "-H", "cuda01", "--", "true"],          # 3
                 ["kill"], ["kill", "2", "--all"], ["kill", "2", "2", "1"], ["kill", "99"],
                 ["clear", "--reset"], ["kill", "--all"], ["clear"], ["list"], ["clear", "--reset"],
                 ["list"], ["submit", "-q", "-g", "0", "-H", "local", "--", "true"]):
        same(*each(pair, *args))


def test_daemon_start_stop_messages(pair):
    for args in (["daemon", "stop"], ["daemon", "status"], ["daemon", "start"], ["daemon", "start"],
                 ["daemon", "status"], ["daemon", "stop"], ["daemon", "stop"], ["daemon", "bogus"]):
        py, rs = each(pair, *args)
        if args[1] == "bogus":                      # argparse and clap word usage errors differently
            assert py.returncode == rs.returncode == 2
            continue
        same(py, rs)
