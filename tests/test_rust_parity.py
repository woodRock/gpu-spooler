"""
The Rust port against the Python version, as black boxes on the fake cluster: the same command,
the same output. Skipped unless GPU_SPOOLER_BIN points at the Rust `task` binary, e.g.

  cargo build --release
  GPU_SPOOLER_BIN=target/release/task .venv/bin/python -m pytest tests/test_rust_parity.py

Each migration step adds the commands it ports.
"""

import json
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
    # Not `list` here: the two homes' daemons share the fake cluster's GPUs, so which GPUs (and how
    # soon) each one's tasks get differs. test_queue_commands compares `list` on one queue.


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


# ── step 4: sync and fetch ────────────────────────────────────────────────────

GIT = ["git", "-c", "user.email=t@example.com", "-c", "user.name=t"]


@pytest.fixture
def clones(pair, fake_cluster):
    """Two identical clones of a repo, same name in different folders (so they share the servers'
    copy), one per home: [(home, root)] for Python, then Rust."""
    import shutil
    import tempfile
    from pathlib import Path
    name = f"proj{int(time.time() * 1000)}"
    roots = [Path(tempfile.mkdtemp(prefix="tsr.", dir="/tmp")) / name for _ in range(2)]
    first = roots[0]
    for rel, text in {"sub/run.py": "print(1)\n", "results/r.txt": "committed\n", "results/keep.txt": "k\n"}.items():
        (first / rel).parent.mkdir(parents=True, exist_ok=True)
        (first / rel).write_text(text)
    subprocess.run(["git", "init", "-q", "-b", "main", str(first)], check=True)
    subprocess.run([*GIT, "-C", str(first), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(first), "commit", "-qm", "init"], check=True)
    shutil.copytree(first, roots[1], symlinks=True)
    yield list(zip(pair, roots)), fake_cluster / "repos" / name
    for r in roots:
        shutil.rmtree(r.parent, ignore_errors=True)
    shutil.rmtree(fake_cluster / "repos" / name, ignore_errors=True)


def in_clones(clones, *args, sub=""):
    """`task ARGS` in each clone: Python in the first, Rust in the second, outputs normalised."""
    (py_side, rs_side), _ = clones
    def one(side, argv):
        home, root = side
        r = subprocess.run([*argv, *args], env=home.env, capture_output=True, text=True, timeout=120,
                           cwd=str(root / sub))
        def clean(s):
            s = re.sub(r"\.task-fetch-[A-Za-z0-9_]+", ".task-fetch-*", s)
            for real, name in ((os.path.realpath(root), "{ROOT}"), (str(root), "{ROOT}"),
                               (os.path.realpath(root.parent), "{PARENT}"), (str(root.parent), "{PARENT}"),
                               (str(home.home), "{HOME}")):
                s = s.replace(real, name)
            return s
        return subprocess.CompletedProcess(r.args, r.returncode, clean(r.stdout), clean(r.stderr))
    return one(py_side, PY), one(rs_side, [os.path.abspath(BIN)])


def both_clones(clones, fn):
    for (_, root) in clones[0]:
        fn(root)


def test_sync_messages(clones):
    _, remote = clones
    for args in (["sync", "--check"], ["sync"], ["sync", "--check"], ["sync", "-H", "nope"], ["sync", "-H", "local"]):
        same(*in_clones(clones, *args))
    same(*in_clones(clones, "sync", "--check", sub="sub"))
    both_clones(clones, lambda root: (root / "sub" / "run.py").write_text("print(2)\n"))   # uncommitted
    same(*in_clones(clones, "sync", "--check"))
    same(*in_clones(clones, "sync"))
    same(*in_clones(clones, "sync", "--check"))
    (remote / "sub" / "run.py").write_text("edited on the server\n")
    same(*in_clones(clones, "sync", "--check"))
    same(*in_clones(clones, "sync"))                   # the same tree: the server's edit is kept
    both_clones(clones, lambda root: (root / "sub" / "run.py").write_text("print(3)\n"))
    same(*in_clones(clones, "sync"))                   # now it conflicts: FAIL, exit 1
    same(*in_clones(clones, "sync", sub="../.."))      # not in a git repo


def test_fetch_messages(clones):
    _, remote = clones
    same(*in_clones(clones, "fetch", "results"))      # no copy on the servers yet
    same(*in_clones(clones, "sync"))
    (remote / "results" / "r.txt").write_text("from the server\n")
    (remote / "results" / "deep").mkdir()
    (remote / "results" / "deep" / "n.txt").write_text("new\n")
    (remote / "results" / "x.bin").write_bytes(bytes(range(256)) * 4)
    os.chmod(remote / "results" / "x.bin", 0o4755)
    os.symlink("r.txt", remote / "results" / "inside_link")
    os.symlink("../../..", remote / "results" / "outside_link")
    os.symlink(str(remote.parent), remote / "results" / "absolute_link")
    for args in (["fetch", "results", "--dry-run"], ["fetch", "results"], ["fetch", "results"],
                 ["fetch", "results/deep/n.txt", "results/missing", "results/keep.txt"]):
        same(*in_clones(clones, *args))
    both_clones(clones, lambda root: (root / "results" / "deep" / "n.txt").write_text("my edit\n"))
    (remote / "results" / "deep" / "n.txt").write_text("newer\n")
    for args in (["fetch", "results"], ["fetch", "results", "--force"], ["fetch", "--dry-run", "results"]):
        same(*in_clones(clones, *args))
    same(*in_clones(clones, "fetch", "../results", sub="sub"))
    both_clones(clones, lambda root: (root / "elsewhere").mkdir())
    for args in ([f"{remote}/results/r.txt:elsewhere"], [f"{remote}/results:copy"], ["results/../.."],
                 ["/tmp"], ["-H", "nope", "results"], ["-H", "local", "results"], ["results", "--after", "999"]):
        same(*in_clones(clones, "fetch", *args))
    for (_, root) in clones[0]:
        assert (root / "results" / "deep" / "n.txt").read_text() == "newer\n"
        assert os.readlink(root / "results" / "inside_link") == "r.txt"
        assert not os.path.lexists(root / "results" / "outside_link")
        assert (root / "results" / "x.bin").stat().st_mode & 0o7777 == 0o755
        assert (root / "elsewhere" / "r.txt").read_text() == "from the server\n"
    py_manifest = json.loads((clones[0][0][0].state / "fetched.json").read_text())
    rs_manifest = json.loads((clones[0][1][0].state / "fetched.json").read_text())
    rel = lambda m, root: sorted(os.path.relpath(k, os.path.realpath(root)) for k in m)
    assert rel(py_manifest, clones[0][0][1]) == rel(rs_manifest, clones[0][1][1])


def test_fetch_after_two_tasks_attributes_each_file(clones):
    """fetch --after A B: each file on the task that wrote it, the shared one reported, and (with
    taura installed) each task's outputs recorded on its own run."""
    import shutil
    sides, remote = clones
    for (_, root) in sides:
        (root / "ledger").mkdir()
        (root / "ledger" / "artifacts.toml").write_text("")
        subprocess.run([*GIT, "-C", str(root), "add", "ledger"], check=True)
        subprocess.run([*GIT, "-C", str(root), "commit", "-qm", "ledger"], check=True)
    same(*in_clones(clones, "sync"))
    a = "sleep 1; mkdir -p results/a && echo a > results/a/x.txt && echo c > results/$(echo common).txt; sleep 2"
    b = "sleep 1; mkdir -p results/b && echo b > results/b/y.txt && echo s > results/shared.txt ; sleep 2"
    for (home, root), argv in zip(sides, (PY, [os.path.abspath(BIN)])):
        # One version at a time: both write the same files in the servers' copy.
        for cmd in (a, b):
            r = subprocess.run([*argv, "submit", "-q", "-g", "1", "-H", "cuda02", "--", cmd], env=home.env,
                               capture_output=True, text=True, cwd=str(root))
            assert r.returncode == 0, r.stderr
        for tid in (1, 2):
            home.wait_status(tid, "done", 60)
    py, rs = in_clones(clones, "fetch", "results", "--after", "1", "2")
    same(py, rs)
    # x.txt is task 1's and y.txt and shared.txt are task 2's (their commands name them); common.txt
    # is named by neither and both ran when it was written.
    assert "Warning: 1 file(s) could not be told apart between tasks 1, 2" in py.stdout, py.stdout
    if shutil.which("taura"):
        assert "Recorded 2 output(s) of task 1" in py.stdout and "Recorded 3 output(s) of task 2" in py.stdout, py.stdout


# ── step 5: tui ───────────────────────────────────────────────────────────────

def test_tui_refuses_a_broken_config(ts):
    bad = ts.home / "bad.toml"
    bad.write_text('[network]\nmode = "sometimes"\n')
    ts.env["GPU_SPOOLER_CONFIG"] = str(bad)
    same(*both(ts, "tui"))
