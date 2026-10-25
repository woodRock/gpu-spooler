#!/usr/bin/env python3
"""
Golden fixtures for the Rust port: run the Python implementation on fixed inputs and write what it
produces to tests/golden/parity/. The Rust tests (crates/gpu-spooler/tests/golden.rs) compare the
port against these files, so everything sent to the servers stays byte for byte the same and the
two versions can share ~/.gpu-spooler/ during the switch-over.

  .venv/bin/python dev/gen_golden.py           # (re)write the fixtures
  .venv/bin/python dev/gen_golden.py --check   # exit 1 if they are out of date (CI)

Each file is JSON: a list of {"name", "input", "output"} cases, or a single text file for long
scripts. Paths under the temporary HOME are written as {HOME} and the ControlMaster directory as
{CM_DIR}; times are formatted in UTC.
"""

import io
import json
import os
import shlex
import shutil
import sys
import tempfile
import time
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "golden" / "parity"

# Before importing gpu_spooler: STATE_DIR is fixed at import from $HOME.
HOME = Path(tempfile.mkdtemp(prefix="gg."))
os.environ["HOME"] = str(HOME)
os.environ["TZ"] = "UTC"
time.tzset()
os.environ.pop("GPU_DETECTIVE_DB", None)
sys.path.insert(0, str(REPO))

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from gpu_spooler import config as C          # noqa: E402
from gpu_spooler import core, daemon, fetch, gpu, gpu_status, jobs, provenance, repo  # noqa: E402
from gpu_spooler.transport import Transport  # noqa: E402

CM_DIR = str(Transport(C.builtin_config()).cm_dir)


def scrub(text: str) -> str:
    return text.replace(CM_DIR, "{CM_DIR}").replace(str(HOME), "{HOME}")


def scrub_json(value):
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, list):
        return [scrub_json(v) for v in value]
    if isinstance(value, tuple):
        return [scrub_json(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub_json(v) for k, v in value.items()}
    return value


FILES: dict[str, str] = {}


def text(name: str, content: str) -> None:
    FILES[name] = scrub(content)


def cases(name: str, rows: list[dict]) -> None:
    FILES[name] = json.dumps(scrub_json(rows), indent=1, ensure_ascii=False) + "\n"


def parsed(src: str) -> C.Config:
    """A config as load_config() returns it for ~/.gpu-spooler/hosts.toml (source set, for headers)."""
    cfg = C.parse_config(tomllib.loads(src))
    cfg.source = HOME / ".gpu-spooler" / "hosts.toml"
    return cfg


def error_of(fn):
    try:
        fn()
    except Exception as e:     # the message is what the port must reproduce
        return {"error": type(e).__name__, "message": str(e)}
    return None


# ── Inputs ─────────────────────────────────────────────────────────────────────

FAKE_CLUSTER_TOML = '''
[network]
mode = "jump"
probe = "127.0.0.1:2211"

[ssh]
identity_file = "~/fake/id_ed25519"
control_persist = "120s"
jump_control_persist = "120s"
extra_options = ["UserKnownHostsFile=~/fake/known_hosts", "IdentitiesOnly=yes", "IdentityAgent = none"]

[jumps.fake]
chain = ["entry", "barretts"]

[hosts.entry]
hostname = "127.0.0.1"
port = 2201
user = "me"

[hosts.barretts]
hostname = "127.0.0.1"
port = 2202
user = "me"

[groups.fake]
user = "me"
hostname = "127.0.0.1"
jump = "fake"
path_map = [ { local = "~/local-src", remote = "/work/remote-src" } ]
remote_root = "/work/repos/"
hosts = ["cuda01", "cuda02"]

[hosts.cuda01]
port = 2211

[hosts.cuda02]
port = 2212
gpu_type = "A40"

[hosts.local]
kind = "local"
cpu_slots = 2
'''

MERGING_TOML = '''
[jumps.j]
chain = ["gate", "u@127.0.0.1:2201", "[::1]:2202", "gw.example.org", "gw.example.org", "gw.other.org"]
[hosts.gate]
hostname = "gate.example"
[groups.a]
user = "alice"
domain = "a.example"
jump = "j"
cpu_slots = 1
tags = ["big"]
hosts = ["x", "y"]
[groups.b]
user = "bob"
tags = ["b2", "big"]
hosts = ["y", "z"]
[hosts.y]
user = "yuser"
hostname = "custom.y"
tags = ["extra", "a"]
enabled = true
gpu_mem_gb = 23.5
[hosts.x]
enabled = false
[hosts.loc]
kind = "local"
cpu_slots = 2
jump = "j"
[limits]
max_gpus = 4
[provenance]
taura = "off"
[detective]
idle_minutes = 45
history = "~/hist.db"
'''

PATHS_TOML = '''
[hosts.h]
remote_root = "/vol/me"
path_map = [
  { local = "/work/j/code", remote = "/vol/a" },
  { local = "/work/j/code/deep/", remote = "/vol/b/" },
  { local = "/mapped", remote = "/elsewhere" },
  { local = "~/data", remote = "/vol/data" },
]
[hosts.root]
path_map = [ { local = "/", remote = "/srv/" } ]
[hosts.plain]
[hosts.loc]
kind = "local"
'''

CONFIG_ERRORS = {
    "bad_toml": "[network\nmode = 1",
    "bad_mode": '[network]\nmode = "sometimes"',
    "bad_taura": '[provenance]\ntaura = "on"',
    "group_unknown_key": '[groups.g]\nhosts = ["a"]\nfoo = 1',
    "host_unknown_key": '[hosts.a]\nbar = 1\nfoo = 2',
    "empty_chain": '[jumps.j]\nchain = []',
    "bad_kind": '[hosts.a]\nkind = "docker"',
    "unknown_jump": '[hosts.a]\njump = "nope"',
    "bad_role": '[hosts.a]\nrole = "boss"',
    "hop_with_jump": '[jumps.j]\nchain = ["a"]\n[hosts.a]\njump = "j"',
    "path_map_entry": '[hosts.a]\npath_map = [ { local = "/x" } ]',
    "limits_unknown": '[limits]\nmax_gpu = 5',
    "max_gpus_zero": '[limits]\nmax_gpus = 0',
    "max_gpus_string": '[limits]\nmax_gpus = "5"',
    "max_gpus_bool": '[limits]\nmax_gpus = true',
    "max_gpus_float": '[limits]\nmax_gpus = 2.5',
    "detective_unknown": '[detective]\nwhen = 1',
    "idle_minutes_zero": '[detective]\nidle_minutes = 0',
    "idle_minutes_bool": '[detective]\nidle_minutes = false',
}

RAW_GPU = """\
GPU_INFO
0, NVIDIA A40, 1, 46067, 46068, 0, Default
1, NVIDIA A40, 300, 45768, 46068, 0, Default
2, NVIDIA A40, 18000, 28068, 46068, 99, Default
3, NVIDIA A40, 2, 46066, 46068, 0, Default
PMON_INFO
# gpu         pid   type     sm    mem    enc    dec    jpg    ofa     fb   command
# Idx           #    C/G      %      %      %      %      %      %     MB   name
    0           -      -      -      -      -      -      -      -      -   -
    1           -      -      -      -      -      -      -      -      -   -
    2       4242      C     99     40      -      -      -      -  17990   python
    3           -      -      -      -      -      -      -      -      -   -
APPS_INFO
2, 4242
3, 5151
PROC_INFO
 4242 alice
 5151 bob
END
"""

RAW_GPU_CASES = {
    "a40_mixed": RAW_GPU,
    "old_pmon_crlf": RAW_GPU.replace("\n", "\r\n").replace(
        "    2       4242      C     99     40      -      -      -      -  17990   python",
        "    2       4242     C    99    40     -     -  17990   python"),
    "empty": "",
    "no_nvidia_smi": "NO_NVIDIA_SMI\n",
    "util_na_and_junk": """\
GPU_INFO
0, NVIDIA RTX A5000, 10, 24554, 24564, [N/A], Default
1, NVIDIA RTX A5000, 23000, 1564, 24564, 87, Exclusive_Process
x, broken, line
2, short
PMON_INFO
    1       777      C     87     40   23000   a_very_long_command_name_that_goes_on_and_on_and_on_for_more_than_sixty_chars
    9       888      C      0      0      5   ghost
APPS_INFO
1, 999
PROC_INFO
  777 carol
END
trailing junk
""",
}


# ── Generators ─────────────────────────────────────────────────────────────────

def gen_shell() -> None:
    words = ["", "plain", "with space", "it's", "a\"b", "$HOME", "~/x", "a*b", "-n", "@%+=:,./-_",
             "naïve", "tab\there", "new\nline", "x;y", "back\\slash", "'"]
    cases("shell_quote.json", [{"name": repr(w), "input": w, "output": shlex.quote(w)} for w in words])
    argvs = [["python3 train.py && echo done"], ["python3", "train.py", "--lr", "3e-4"],
             ["bash", "-c", "echo $HOME"], ["echo", "it's", ""], ["./run.sh", "--name", "a b"]]
    cases("command_from_argv.json", [{"name": " ".join(a), "input": a, "output": core.command_from_argv(a)}
                                     for a in argvs])
    paths = ["~", "~/", "~/x y/z", "/vol/me/proj", "/path/it's", "~user/x", "relative dir"]
    cases("sh_path.json", [{"name": p, "input": p, "output": repo.sh_path(p)} for p in paths])


def gen_fmt() -> None:
    durs = [None, -1, 0, 0.9, 59, 59.99, 60, 61, 3599, 3600, 3661, 86400 * 2 + 7]
    cases("fmt_duration.json", [{"name": str(d), "input": d, "output": core.fmt_duration(d)} for d in durs])
    ts = [None, 0, 1761400000.5, 1735689599.9]
    cases("fmt_ts.json", [{"name": str(t), "input": t, "output": core.fmt_ts(t)} for t in ts])
    cases("truncate.json", [{"name": f"{s}/{n}", "input": [s, n], "output": core.truncate(s, n)}
                            for s, n in [("short", 10), ("exactly10!", 10), ("this is too long", 10),
                                         ("ünïcödé text", 5), ("", 3)]])
    cases("one_line.json", [{"name": repr(s), "input": s, "output": core.one_line(s)}
                            for s in ["a\nb", "  lots   of\t\tspace  ", "", "x\r\ny"]])
    names = ["NVIDIA A40", "NVIDIA  RTX A5000", "nvidia l4", "Quadro RTX 6000", "  NVIDIA   RTX 6000 Ada Generation ",
             "NVIDIA", "NVIDIAX"]
    cases("gpu_model.json", [{"name": n, "input": n, "output": gpu.gpu_model(n)} for n in names])
    matches = [(None, "anything"), ("", "anything"), ("rtx a5000", "NVIDIA RTX A5000"),
               ("NVIDIA RTX A5000", "RTX A5000"), ("RTX 6000", "NVIDIA RTX 6000 Ada Generation"),
               ("RTX 6000", "Quadro RTX 6000"), ("A40, RTX A5000", "NVIDIA A40"), (" , ", "A40"),
               ("RTX A5000", "NVIDIA RTX PRO 5000 Blackwell")]
    cases("gpu_type_matches.json", [{"name": f"{s!r} ~ {n}", "input": [s, n], "output": gpu.gpu_type_matches(s, n)}
                                    for s, n in matches])


def _row(**kw) -> dict:
    base = {"id": 7, "status": "done", "exit_code": None, "started_at": None, "finished_at": None}
    base.update(kw)
    return base


def gen_wait() -> None:
    rows = [_row(status="done", exit_code=0, started_at=1000.0, finished_at=1221.0),
            _row(status="failed", exit_code=3, started_at=1000.0, finished_at=1005.5),
            _row(status="failed", exit_code=0), _row(status="failed", exit_code=None),
            _row(status="killed", started_at=1000.0, finished_at=4700.0), _row(status="cancelled"),
            _row(status="lost", started_at=1000.0, finished_at=1100.0)]
    cases("wait_exit_code.json", [{"name": f"{r['status']}/{r['exit_code']}", "input": r,
                                   "output": {"code": core.wait_exit_code(r), "describe": core.describe_end(r)}}
                                  for r in rows])


def _config_json(cfg: C.Config) -> dict:
    d = asdict(cfg)
    d.pop("source")
    d.pop("mtime")
    d["hosts"] = [asdict(h) for h in cfg.hosts.values()]    # a list keeps the order
    d["workers"] = [h.name for h in cfg.workers()]
    return d


def gen_config() -> None:
    rows = [{"name": "builtin", "input": None, "output": _config_json(C.builtin_config())}]
    for name, src in [("template", C.config_template("woodj4")), ("fake_cluster", FAKE_CLUSTER_TOML),
                      ("merging", MERGING_TOML), ("paths", PATHS_TOML), ("empty", "")]:
        rows.append({"name": name, "input": src, "output": _config_json(C.parse_config(tomllib.loads(src)))})
    for name, src in CONFIG_ERRORS.items():
        def load(src=src):
            try:
                data = tomllib.loads(src)
            except tomllib.TOMLDecodeError:
                raise C.ConfigError("invalid TOML") from None   # the decoder's own wording may differ
            return C.parse_config(data)
        rows.append({"name": name, "input": src, "output": error_of(load)})
    cases("parse_config.json", rows)
    text("config_template.toml", C.config_template("woodj4"))

    cfg = C.parse_config(tomllib.loads(MERGING_TOML))
    specs = [None, "", "b", "big", "loc, z", "a,b", "y,,y", "x", "nope", "z,nope"]
    sel = []
    for s in specs:
        err = error_of(lambda s=s: cfg.select(s))
        sel.append({"name": repr(s), "input": s,
                    "output": err or [h.name for h in cfg.select(s)]})
    cases("select.json", [{"name": "merging", "input": MERGING_TOML, "output": None}] + sel)

    cfg = C.parse_config(tomllib.loads(PATHS_TOML))
    probes = [("h", "/work/j/code"), ("h", "/work/j/code/"), ("h", "/work/j/code/x/y"), ("h", "/work/j/codex"),
              ("h", "/work/j/code/deep"), ("h", "/work/j/code/deep/more"), ("h", "/mapped/x"),
              ("h", str(HOME / "data" / "set")), ("h", "/unmapped/p"), ("root", "/"), ("root", "/a/b"),
              ("plain", "/work/j/code")]
    cases("map_path.json", [{"name": f"{h}:{p}", "input": [PATHS_TOML, h, p],
                             "output": {"map_path": cfg.map_path(cfg.get(h), p),
                                        "is_mapped": cfg.is_mapped(cfg.get(h), p)}} for h, p in probes])
    rd = [("h", "/work/proj/a/b", "/work/proj"), ("h", "/work/proj", "/work/proj"),
          ("h", "/work/proj/", "/work/proj/"), ("h", "/mapped/x", "/mapped"), ("h", "/work/proj", None),
          ("plain", "/work/proj", "/work/proj"), ("loc", "/work/proj", "/work/proj"),
          ("h", "/work/j/code/sub", "/work/j/code")]
    cases("remote_dir.json", [{"name": f"{h}:{w}:{r}", "input": [PATHS_TOML, h, w, r],
                               "output": cfg.remote_dir(cfg.get(h), w, r)} for h, w, r in rd])


def gen_ssh_config() -> None:
    def render(src, mode):
        cfg = C.builtin_config() if src is None else parsed(src)
        return Transport(cfg).render_ssh_config(mode)

    template = C.config_template("woodj4")
    for name, src, mode in [("builtin.direct", None, "direct"), ("template.direct", template, "direct"),
                            ("template.jump", template, "jump"), ("fake_cluster.jump", FAKE_CLUSTER_TOML, "jump"),
                            ("fake_cluster.direct", FAKE_CLUSTER_TOML, "direct"),
                            ("merging.jump", MERGING_TOML, "jump")]:
        text(f"ssh_config/{name}", render(src, mode))
    # With a ~/.ssh/config to Include (last, so the files above have none).
    (HOME / ".ssh").mkdir()
    (HOME / ".ssh" / "config").write_text("Host *\n")
    text("ssh_config/builtin.include", render(None, "direct"))


def gen_jobs() -> None:
    launches = [
        ("plain", ("12-a1b2c3", "/vol/me/proj", "0", "python3 train.py --epochs 5", False, None)),
        ("multi_gpu_delay", ("3-000001", "/vol/me/proj", "1,2", "torchrun --nproc 2 t.py", True, None)),
        ("home_path", ("4-ffffff", "~/work dir", "", "echo 'quoted' && ls \"$HOME\"", False, None)),
        ("tilde_only", ("5-abcdef", "~", "0", "true", False, None)),
        ("probe", ("6-123456", "/vol/me/it's", "0", "PYTHONPATH=src python3 -m x",
                   False, provenance.probe_command("PYTHONPATH=src python3 -m x"))),
        ("multiline", ("7-654321", "/x", "3", "for i in 1 2; do\n  echo $i\ndone", False, None)),
    ]
    rows = []
    for name, args in launches:
        text(f"jobs/launch.{name}.sh", jobs.launch_script(*args, ) if args[5] is None
             else jobs.launch_script(*args[:5], probe=args[5]))
        rows.append({"name": name, "input": {"key": args[0], "wdir": args[1], "gpus": args[2], "command": args[3],
                                             "delay": args[4], "probe": args[5]},
                     "output": f"jobs/launch.{name}.sh"})
    cases("jobs_launch.json", rows)

    text("jobs/status.sh", jobs.status_script([("1-aaaaaa", 0), ("22-bbbbbb", 1048576), ("3-c'c", 5)]))
    text("jobs/kill.sh", jobs.kill_script("12-a1b2c3"))
    text("jobs/kill.grace3.sh", jobs.kill_script("12-a1b2c3", grace=3))
    text("jobs/cleanup.sh", jobs.cleanup_script([1, 22, 333]))
    cases("jobs_scripts.json", [
        {"name": "status", "input": [["1-aaaaaa", 0], ["22-bbbbbb", 1048576], ["3-c'c", 5]], "output": "jobs/status.sh"},
        {"name": "kill", "input": ["12-a1b2c3", 10], "output": "jobs/kill.sh"},
        {"name": "kill.grace3", "input": ["12-a1b2c3", 3], "output": "jobs/kill.grace3.sh"},
        {"name": "cleanup", "input": [1, 22, 333], "output": "jobs/cleanup.sh"},
    ])

    def status(out):
        st = jobs.parse_status(out)
        return None if st is None else {k: {"state": v.state, "rc": v.rc, "size": v.size,
                                            "data": v.data.decode("latin-1")} for k, v in st.items()}
    outs = {
        "complete": ("@@JOB 1-a\nSTATE running\nSIZE 12\nDATA aGVsbG8gd29ybGQK\n@@JOB 2-b\nSTATE done 3\nSIZE 0\n"
                     "DATA \n@@JOB 3-c\nSTATE missing\nSIZE 0\nDATA\n@@JOB 4-d\nSTATE gone\nSIZE \nDATA \n"
                     "@@JOB 5-e\nSTATE done\nSIZE x\nDATA \n@@JOB 6-f\nSTATE done oops\nSIZE 1\nDATA \n@@END\n"),
        "cut_short": "@@JOB 1-a\nSTATE running\nSIZE 12\nDATA aGVsbG8=\n",
        "bad_base64": "@@JOB 1-a\nSTATE running\nSIZE 1\nDATA !!!notbase64\n@@END\n",
        "noise_before_job": "Welcome to cuda01\nSTATE done 0\n@@JOB 1-a\nSTATE done 0\nSIZE 0\nDATA \n@@END\nafter\n",
        "empty": "",
        "only_end": "@@END\n",
    }
    cases("jobs_parse_status.json", [{"name": n, "input": o, "output": status(o)} for n, o in outs.items()])

    launch_outs = {
        "pid_and_git": "TS_GIT 0123456789abcdef0123456789abcdef01234567 2\nTS_DIRTY src/a.py\n"
                       "TS_DIRTY \"we ird.py\"\nTS_DIRTY old.py -> new.py\nTS_PID 4242\n",
        "pid_only": "noise\nTS_PID 17\n",
        "bad_pid": "TS_PID abc\nTS_PID 5\n",
        "git_bad_count": "TS_GIT abc x\nTS_PID 1\n",
        "nothing": "",
    }
    cases("jobs_parse_launch.json", [{"name": n, "input": o,
                                      "output": {"pid": jobs.parse_launch(o), "git": jobs.parse_git(o),
                                                 "dirty": jobs.parse_dirty(o)}} for n, o in launch_outs.items()])


def gen_gpu() -> None:
    text("gpu/remote_cmd.sh", gpu.REMOTE_CMD)
    rows = []
    for name, raw in RAW_GPU_CASES.items():
        gpus = gpu.parse_output(raw)
        rows.append({"name": name, "input": raw, "output": [
            dict(asdict(g), is_free=g.is_free, mem_total_gb=g.mem_total_gb) for g in gpus]})
    cases("gpu_parse_output.json", rows)


def _results() -> dict[str, gpu.ServerResult]:
    a40 = gpu.parse_output(RAW_GPU)
    a5000 = gpu.parse_output(RAW_GPU_CASES["util_na_and_junk"])
    return {
        "cuda02": gpu.ServerResult("cuda02", True, gpus=a40),
        "cuda01": gpu.ServerResult("cuda01", True, gpus=a5000),
        "cuda14": gpu.ServerResult("cuda14", True, gpus=[
            gpu.GpuInfo(i, "NVIDIA A40", 1, 46067, 46068, 0) for i in range(4)]),
        "cuda20": gpu.ServerResult("cuda20", False, error="ssh: connect to host cuda20 port 22: Connection refused"
                                   + " and a long tail so the error is cut at sixty characters for the table"),
        "local": gpu.ServerResult("local", True),
    }


def gen_snapshot() -> None:
    res = _results()
    cases("server_result.json", [{"name": n, "input": asdict(r),
                                  "output": {"total_gpus": r.total_gpus, "free_gpus": r.free_gpus,
                                             "busy_gpus": r.busy_gpus, "total_vram_mb": r.total_vram_mb,
                                             "free_vram_mb": r.free_vram_mb, "models": r.models}}
                                 for n, r in res.items()])


DAEMON_TOML = '''
[groups.g]
hosts = ["cuda01", "cuda02", "cuda14", "cuda20"]
[hosts.cuda14]
gpu_mem_gb = 48
[hosts.local]
kind = "local"
cpu_slots = 2
'''


def gen_daemon() -> None:
    cfg = C.parse_config(tomllib.loads(DAEMON_TOML))
    res = _results()
    running = [{"server": "cuda14", "gpu_indices": "0,1", "num_gpus": 2},
               {"server": "local", "gpu_indices": "", "num_gpus": 0},
               {"server": "gone", "gpu_indices": "0", "num_gpus": 1}]

    def q(i, n, **kw):
        return {"id": i, "num_gpus": n, "min_mem": kw.get("min_mem", 0), "host_spec": kw.get("host_spec"),
                "gpu_type": kw.get("gpu_type")}
    queues = {
        "fill_small_first": [q(1, 1), q(2, 1), q(3, 2), q(4, 1)],
        "cpu_jobs": [q(1, 0), q(2, 0), q(3, 0, host_spec="cuda01")],
        "min_mem": [q(1, 1, min_mem=40), q(2, 1, min_mem=100)],
        "gpu_type": [q(1, 1, gpu_type="A40"), q(2, 1, gpu_type="rtx a5000"), q(3, 1, gpu_type="L4")],
        "host_spec": [q(1, 1, host_spec="cuda14"), q(2, 1, host_spec="nope"), q(3, 4, host_spec="g")],
        "too_big": [q(1, 5), q(2, 1)],
    }
    rows = []
    for name, queue in queues.items():
        for budget in (None, 2):
            states = daemon.host_states(cfg, res, running)
            before = {n: {"free": {str(k): v for k, v in s.free.items()}, "cpu_free": s.cpu_free,
                          "mem_gb": s.mem_gb} for n, s in states.items()}
            out = daemon.assign_tasks(queue, states, cfg, budget)
            rows.append({"name": f"{name}/budget={budget}", "input": {"queue": queue, "budget": budget},
                         "output": {"states": before, "assignments": [list(a) for a in out]}})
    cases("daemon_assign.json", [{"name": "setup", "input": {"config": DAEMON_TOML,
                                                              "results": {n: asdict(r) for n, r in res.items()},
                                                              "running": running}, "output": None}] + rows)

    class Cfg:
        def __init__(self, m):
            self.max_gpus = m
    cases("gpu_budget.json", [{"name": f"{m}", "input": m, "output": daemon.gpu_budget(Cfg(m), running)}
                              for m in (None, 1, 2, 5)])


def gen_repo() -> None:
    for name, path in [("plain", "/vol/me/proj"), ("home", "~/repos/my proj"), ("quote", "/vol/it's")]:
        text(f"repo/check.{name}.sh", repo.check_script(path))
        text(f"repo/prepare.{name}.sh", repo._prepare_script(path))
        text(f"repo/checkout.{name}.sh", repo._checkout_script(path, "0123456789abcdef0123456789abcdef01234567"))
    cases("repo_constants.json", [{"name": "FIND_GIT", "input": None, "output": repo.FIND_GIT},
                                  {"name": "RECEIVE_PACK", "input": None, "output": repo.RECEIVE_PACK},
                                  {"name": "SYNC_REF", "input": None, "output": repo.SYNC_REF}])


def gen_fetch() -> None:
    text("fetch/list.sh", fetch._py_script(fetch._LIST_PY, ["/vol/me/proj/results", "~/x y", "/vol/it's"]))
    text("fetch/tar.sh", fetch._py_script(fetch._TAR_PY, {"base": "/vol/me/proj/results",
                                                          "members": ["a.csv", "sub/b c.txt", "link"]}))


def gen_provenance() -> None:
    text("provenance/probe.py", provenance.PROBE)
    cmds = ["python3 train.py", "PYTHONPATH=src CUDA=1 python3.11 -m pkg", "/usr/bin/python t.py",
            "torchrun --nproc 2 t.py", "bash -c 'python3 x.py'", "A=1", "", "python3 'unterminated",
            "env X=1 python3 t.py", "./venv/bin/python3.10 run.py", "pythonista x"]
    cases("probe_command.json", [{"name": c, "input": c, "output": provenance.probe_command(c)} for c in cmds])
    rows = [{"id": 12, "submitted_at": 1761400000.5, "retry_count": 0},
            {"id": 12, "submitted_at": 1761400000.5, "retry_count": None},
            {"id": 3, "submitted_at": 1735689599.9, "retry_count": 2}]
    cases("run_id.json", [{"name": f"{r['id']}/{r['retry_count']}", "input": r, "output": provenance.run_id(r)}
                          for r in rows])


GPU_STATUS_TOML = '''
[groups.g]
hosts = ["cuda01", "cuda02", "cuda14", "cuda20"]
[hosts.cuda01]
gpu_type = "RTX A5000"
[hosts.cuda02]
gpu_type = "RTX A6000"
[hosts.local]
kind = "local"
gpu_type = "none"
cpu_slots = 2
'''


def gen_gpu_status() -> None:
    cfg = parsed(GPU_STATUS_TOML)
    results = list(_results().values())

    def capture(fn, *args, **kw) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            fn(*args, **kw)
        return buf.getvalue()

    rows = []

    def table(name, fn, *args, **kw):
        text(f"gpu_status/{name}.txt", capture(fn, *args, **kw))
        rows.append({"name": name, "input": None, "output": f"gpu_status/{name}.txt"})

    for free in (False, True):
        sfx = ".free" if free else ""
        table(f"summary{sfx}", gpu_status.print_summary_table, results, cfg, free_only=free)
        table(f"detail{sfx}", gpu_status.print_detail_table, results, free_only=free)
    table("label_warnings", gpu_status.print_label_warnings, results, cfg)
    table("totals", gpu_status.print_totals, results)
    table("totals.unreachable0", gpu_status.print_totals, [r for r in results if r.reachable])
    cases("gpu_status.json", [{"name": "setup", "input": {"config": GPU_STATUS_TOML,
                                                          "results": [asdict(r) for r in results]},
                               "output": None}] + rows)


def _task(**kw) -> dict:
    """A tasks row with every column, as the database returns it."""
    row = {"id": 12, "command": "python3 train.py --epochs 5", "num_gpus": 1, "min_mem": 0, "status": "queued",
           "server": None, "gpu_indices": None, "ssh_pid": None, "submitted_at": 1761400000.5, "started_at": None,
           "finished_at": None, "exit_code": None, "log_file": None, "working_dir": "/vol/me/proj", "name": None,
           "retry_count": 0, "job_key": None, "remote_pid": None, "log_offset": 0, "host_spec": None,
           "remote_dir": None, "last_seen": None, "repo_root": None, "gpu_type": None, "git_commit": None,
           "git_dirty": None, "git_dirty_paths": None, "no_record": 0}
    row.update(kw)
    return row


TASK_DETAILS_TOML = '''
[hosts.cuda01]
gpu_type = "RTX A5000"
gpu_mem_gb = 24
[hosts.cuda05]
gpu_mem_gb = 47.5
[hosts.cuda06]
[hosts.local]
kind = "local"
cpu_slots = 2
'''

TASK_SNAPSHOT = {"ts": 1761400000.0, "mode": "direct", "hosts": {
    "cuda02": {"name": "cuda02", "reachable": True, "error": "", "ts": 1761400000.0, "gpus": [
        {"index": 0, "name": "NVIDIA A40", "mem_used_mb": 1, "mem_free_mb": 1, "mem_total_mb": 2,
         "utilization_pct": 0, "processes": []},
        {"index": 1, "name": "NVIDIA L4", "mem_used_mb": 1, "mem_free_mb": 1, "mem_total_mb": 2,
         "utilization_pct": 0, "processes": []},
        {"index": 2, "name": "NVIDIA A40", "mem_used_mb": 1, "mem_free_mb": 1, "mem_total_mb": 2,
         "utilization_pct": 0, "processes": []}]}}}


def gen_task_details() -> None:
    cfg = parsed(TASK_DETAILS_TOML)
    snap = HOME / "snapshot.json"
    snap.write_text(json.dumps(TASK_SNAPSHOT))
    core.set_config(cfg)
    core.SNAPSHOT_PATH = snap
    rows = {
        "queued_minimal": _task(),
        "queued_cpu_everything": _task(num_gpus=0, name="plots", host_spec="local", min_mem=40, gpu_type="A40",
                                       repo_root="/work/proj", remote_dir="/vol/me/proj/sub",
                                       working_dir="/work/proj/sub"),
        "done_snapshot_models": _task(status="done", server="cuda02", gpu_indices="0,2", started_at=1761400010.0,
                                      finished_at=1761400231.9, exit_code=0, job_key="12-a1b2c3", remote_pid=4242,
                                      log_file="{HOME}/.gpu-spooler/logs/12.log", retry_count=2,
                                      remote_dir="/vol/me/proj"),
        "done_all_models": _task(status="done", server="cuda02", gpu_indices=" , x", started_at=1761400010.0,
                                 finished_at=1761400011.0, exit_code=0),
        "failed_configured_type": _task(status="failed", server="cuda01", gpu_indices="1", started_at=1761400010.0,
                                        finished_at=1761407300.0, exit_code=3, job_key="12-ffffff"),
        "configured_mem_only": _task(status="killed", server="cuda05", gpu_indices="0", started_at=1761400010.0,
                                     finished_at=1761400070.0),
        "configured_nothing": _task(status="lost", server="cuda06", gpu_indices="0", started_at=1761400010.0,
                                    finished_at=1761400070.0, exit_code=None),
        "unknown_server": _task(status="cancelled", server="gone", finished_at=1761400070.0),
    }
    out = []
    for name, row in rows.items():
        out.append({"name": name, "input": row, "output": [list(x) for x in core.task_details(row)]})
    cases("task_details.json", [{"name": "setup", "input": {"config": TASK_DETAILS_TOML, "snapshot": TASK_SNAPSHOT},
                                 "output": None}] + out)
    core.set_config(C.builtin_config())


def gen_format_g() -> None:
    vals = [0, 0.5, 1, 1.5, 5, 10, 59.99, 100, 3600, 123456, 999999, 1000000, 1234567, 0.0001, 0.00001,
            1e-7, 2.5e10, 0.1 + 0.2, 7.123456789, 86400.0]
    cases("format_g.json", [{"name": repr(v), "input": v, "output": f"{v:g}"} for v in vals])


def gen_tar_filter() -> None:
    """tarfile's 'data' filter, which fetch relies on, over a destination with a symlink that
    escapes it (dest/out → {HOME}/elsewhere) and one that stays inside (dest/in → sub). Resolved
    paths are written as {DEST}, {HOME} (the generator's home) and {TMP} (the folder above it)."""
    import tarfile
    dest = HOME / "dest"
    (dest / "sub").mkdir(parents=True)
    (HOME / "elsewhere").mkdir()
    os.symlink(str(HOME / "elsewhere"), dest / "out")
    os.symlink("sub", dest / "in")
    real, real_home, real_tmp = (os.path.realpath(p) for p in (dest, HOME, HOME.parent))
    T = {"file": tarfile.REGTYPE, "dir": tarfile.DIRTYPE, "symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE,
         "fifo": tarfile.FIFOTYPE, "chr": tarfile.CHRTYPE}
    members = [
        ("plain", "file", 0o644, ""), ("exec", "file", 0o775, ""), ("noexec_group_x", "file", 0o651, ""),
        ("setuid", "file", 0o4755, ""), ("ro", "file", 0o400, ""), ("deep/a/b.txt", "file", 0o600, ""),
        ("/abs/x", "file", 0o644, ""), ("//abs2", "file", 0o644, ""), ("../up", "file", 0o644, ""),
        ("a/../../up2", "file", 0o644, ""), ("out/etc/passwd", "file", 0o644, ""), ("in/ok.txt", "file", 0o644, ""),
        ("d", "dir", 0o777, ""), ("ln", "symlink", 0o777, "plain"), ("ln_sub", "symlink", 0o777, "../plain"),
        ("sub/ln_up", "symlink", 0o777, "../plain"), ("ln_out", "symlink", 0o777, "../../etc"),
        ("ln_abs", "symlink", 0o777, "/etc/passwd"), ("ln_via_out", "symlink", 0o777, "out/etc"),
        ("ln_via_in", "symlink", 0o777, "in/x"), ("hl", "hardlink", 0o644, "plain"),
        ("hl_out", "hardlink", 0o644, "../x"), ("hl_abs", "hardlink", 0o644, "/x"), ("fifo", "fifo", 0o644, ""),
        ("dev", "chr", 0o644, ""), ("it's", "symlink", 0o777, "../../x"),
    ]
    rows = []
    for name, kind, mode, link in members:
        ti = tarfile.TarInfo(name)
        ti.type, ti.mode, ti.linkname = T[kind], mode, link
        try:
            new = tarfile.data_filter(ti, str(dest))
            out = {"name": new.name, "mode": new.mode}
        except tarfile.FilterError as e:
            out = {"error": type(e).__name__,
                   "message": str(e).replace(real, "{DEST}").replace(real_home, "{HOME}").replace(real_tmp, "{TMP}")}
        rows.append({"name": f"{kind} {name}", "input": {"name": name, "kind": kind, "mode": mode, "link": link},
                     "output": out})
    cases("tar_filter.json", rows)


def _segments(t) -> list:
    """A rich Text as [text, style] runs, unstyled gaps as ''."""
    marks = sorted({0, len(t.plain), *[x for sp in t.spans for x in (sp.start, sp.end)]})
    out = []
    for a, b in zip(marks, marks[1:]):
        styles = [str(sp.style) for sp in t.spans if sp.start <= a and b <= sp.end]
        if a < b:
            out.append([t.plain[a:b], " ".join(styles)])
    return out


def gen_tui() -> None:
    from gpu_spooler import tui
    rows = [_task(id=3, status=st, num_gpus=n, host_spec=hs, server=sv, gpu_indices=gi, name=nm,
                  started_at=1761400010.0 if st != "queued" else None,
                  finished_at=1761400231.0 if st not in ("queued", "running") else None,
                  command="python3 train.py\n  --epochs 5")
            for st, n, hs, sv, gi, nm in [("queued", 1, None, None, None, None), ("done", 2, "g", "cuda02", "0,2", "lbl"),
                                          ("failed", 0, "local", "local", None, ""), ("killed", 1, None, "x", "1", None),
                                          ("cancelled", 1, None, None, None, None), ("lost", 1, None, "y", "0", None),
                                          ("weird", 1, None, None, None, None)]]
    cases("tui_task_row.json", [{"name": r["status"], "input": r,
                                  "output": [str(c) if not hasattr(c, "spans") else [c.plain, str(c.style)]
                                             for c in tui.task_row(r)]} for r in rows])
    snapshot = {"ts": 1761400000.0, "mode": "jump", "hosts": {
        "cuda01": {"name": "cuda01", "reachable": True, "error": "", "gpus": [
            {"index": 0, "name": "NVIDIA RTX A5000", "mem_used_mb": 9000, "mem_free_mb": 0, "mem_total_mb": 24564,
             "utilization_pct": 90, "processes": [{"pid": 1, "user": "?", "mem_used_mb": 1, "command": "x"},
                                                  {"pid": 2, "user": "alice_long", "mem_used_mb": 1, "command": "y"}]},
            {"index": 1, "name": "NVIDIA RTX A5000", "mem_used_mb": 1, "mem_free_mb": 0, "mem_total_mb": 24564,
             "utilization_pct": 0, "processes": []},
            {"index": 2, "name": "NVIDIA RTX A5000", "mem_used_mb": 1, "mem_free_mb": 0, "mem_total_mb": 24564,
             "utilization_pct": 0, "processes": []},
            {"index": 3, "name": "NVIDIA RTX A5000", "mem_used_mb": 300, "mem_free_mb": 0, "mem_total_mb": 24564,
             "utilization_pct": 0, "processes": []}]},
        "zz-extra": {"name": "zz-extra", "reachable": True, "error": "", "gpus": []},
        "cuda02": {"name": "cuda02", "reachable": False, "error": "ssh: connect to host cuda02 port 22: Connection "
                   "timed out after a long while and more text", "gpus": []},
        "cuda03": {"name": "cuda03", "reachable": False, "error": "", "gpus": []}}}
    running = [{"id": 12, "server": "cuda01", "gpu_indices": "1,3"}, {"id": 4, "server": "cuda02", "gpu_indices": "0"},
               {"id": 5, "server": "cuda01", "gpu_indices": " , x"}]
    workers = ["cuda03", "cuda01", "cuda02", "missing"]
    out = []
    for name, snap, daemon in [("full", snapshot, True), ("none", None, True), ("none_daemon_stopped", None, False),
                               ("no_hosts", {"ts": 1, "hosts": {}}, False)]:
        orig = core.daemon_is_running
        core.daemon_is_running = lambda d=daemon: d
        try:
            t = tui.render_gpu_map(snap, running, workers, now=1761400065.0)
        finally:
            core.daemon_is_running = orig
        out.append({"name": name, "input": {"snapshot": snap, "running": running, "workers": workers,
                                            "now": 1761400065.0, "daemon": daemon}, "output": _segments(t)})
    cases("tui_gpu_map.json", out)


def main() -> int:
    check = "--check" in sys.argv[1:]
    try:
        for gen in (gen_shell, gen_fmt, gen_wait, gen_config, gen_ssh_config, gen_jobs, gen_gpu, gen_snapshot,
                    gen_daemon, gen_repo, gen_fetch, gen_provenance, gen_gpu_status, gen_task_details, gen_format_g, gen_tar_filter, gen_tui):
            gen()
    finally:
        shutil.rmtree(HOME, ignore_errors=True)

    index = "".join(f"{name}\n" for name in sorted(FILES))
    FILES["INDEX"] = index
    if check:
        stale = [n for n, body in FILES.items()
                 if not (OUT / n).is_file() or (OUT / n).read_text(encoding="utf-8") != body]
        extra = [str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()
                 and str(p.relative_to(OUT)) not in FILES] if OUT.is_dir() else []
        if stale or extra:
            print("golden fixtures are out of date; run: python dev/gen_golden.py", file=sys.stderr)
            for n in stale + extra:
                print(f"  {n}", file=sys.stderr)
            return 1
        print(f"{len(FILES)} golden fixtures up to date")
        return 0
    if OUT.is_dir():
        shutil.rmtree(OUT)
    for name, body in FILES.items():
        path = OUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    print(f"wrote {len(FILES)} files to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
