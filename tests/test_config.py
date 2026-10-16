import os
import sys

import pytest

from gpu_spooler import config as C

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def parse(text: str) -> C.Config:
    return C.parse_config(tomllib.loads(text))


def test_missing_file_gives_builtin(tmp_path):
    cfg = C.load_config(tmp_path / "nope.toml")
    assert cfg.builtin and cfg.source is None
    assert cfg.mode == "direct"
    assert [h.name for h in cfg.workers()] == C.DEFAULT_SERVERS
    assert len(cfg.workers()) == 30
    h = cfg.get("cuda14")
    assert (h.hostname, h.user, h.jump, h.kind) == (None, None, None, "ssh")
    assert (h.gpu_type, h.num_gpus, h.gpu_mem_gb) == ("A40", 4, 48)
    assert not any(h.is_local for h in cfg.hosts.values())


def test_env_var_overrides_path(tmp_path, monkeypatch):
    p = tmp_path / "h.toml"
    p.write_text('[hosts.box]\nhostname = "box.example"\n')
    monkeypatch.setenv("TASK_SPOOLER_CONFIG", str(p))
    cfg = C.load_config()
    assert cfg.source == p
    assert [h.name for h in cfg.workers()] == ["box"]
    assert cfg.mode == "auto"


def test_template_parses_and_covers_all_servers():
    cfg = parse(C.config_template("someone"))
    assert cfg.mode == "auto"
    assert [h.name for h in cfg.workers()] == C.DEFAULT_SERVERS
    h = cfg.get("cuda14")
    assert h.hostname == "cuda14.ecs.vuw.ac.nz"
    assert h.user == "someone"
    assert h.jump == "ecs"
    assert h.gpu_mem_gb == 48 and h.cpu_cores == 64
    hops = cfg.chain(h)
    assert [x.hostname for x in hops] == ["entry.ecs.vuw.ac.nz", "barretts.ecs.vuw.ac.nz"]
    assert [x.name for x in hops] == ["ts-entry", "ts-barretts"]
    assert all(x.role == "jump" and x.user == "someone" for x in hops)


def test_raw_specs_ports_and_alias_dedup():
    cfg = parse('''
[jumps.j]
chain = ["u@127.0.0.1:2201", "v@127.0.0.1:2202", "gw.example.org"]
[hosts.w]
jump = "j"
''')
    hops = cfg.chain(cfg.get("w"))
    assert [(h.user, h.hostname, h.port) for h in hops] == [
        ("u", "127.0.0.1", 2201), ("v", "127.0.0.1", 2202), (None, "gw.example.org", None)]
    assert len({h.alias for h in hops}) == 3
    assert hops[2].alias == "ts-gw"
    assert [h.name for h in cfg.workers()] == ["w"]


def test_named_hop_and_group_merging():
    cfg = parse('''
[jumps.j]
chain = ["gate"]
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
hosts = ["y", "z"]
[hosts.y]
user = "yuser"
hostname = "custom.y"
enabled = true
[hosts.x]
enabled = false
[hosts.loc]
kind = "local"
cpu_slots = 2
''')
    assert cfg.get("gate").role == "jump"
    y = cfg.get("y")
    assert (y.user, y.hostname, y.jump, y.cpu_slots) == ("yuser", "custom.y", "j", 1)
    assert y.tags == ["a", "big", "b"] and y.group == "a"
    z = cfg.get("z")
    assert (z.user, z.hostname, z.group) == ("bob", None, "b")
    assert [h.name for h in cfg.workers()] == ["y", "z", "loc"]
    assert cfg.get("loc").is_local and cfg.get("loc").cpu_slots == 2

    assert [h.name for h in cfg.select(None)] == ["y", "z", "loc"]
    assert [h.name for h in cfg.select("b")] == ["y", "z"]
    assert [h.name for h in cfg.select("big")] == ["y"]
    assert [h.name for h in cfg.select("loc, z")] == ["z", "loc"]
    with pytest.raises(C.ConfigError):
        cfg.select("x")            # disabled
    with pytest.raises(C.ConfigError):
        cfg.select("nope")


def test_map_path_longest_prefix():
    cfg = parse('''
[hosts.h]
path_map = [
  { local = "/Users/j/code", remote = "/vol/a" },
  { local = "/Users/j/code/deep/", remote = "/vol/b/" },
]
''')
    h = cfg.get("h")
    assert cfg.map_path(h, "/Users/j/code") == "/vol/a"
    assert cfg.map_path(h, "/Users/j/code/proj/x") == "/vol/a/proj/x"
    assert cfg.map_path(h, "/Users/j/code/deep/q") == "/vol/b/q"
    assert cfg.map_path(h, "/Users/j/codex") == "/Users/j/codex"
    assert cfg.map_path(h, "/tmp") == "/tmp"
    assert cfg.is_mapped(h, "/Users/j/code/proj")
    tmp = os.path.realpath("/tmp")             # /private/tmp on macOS
    cfg2 = parse('[hosts.h]\npath_map = [ { local = "/tmp", remote = "/r" } ]')
    assert cfg2.map_path(cfg2.get("h"), tmp + "/x") == "/r/x"
    assert not cfg.is_mapped(h, "/tmp")


@pytest.mark.parametrize("bad", [
    '[network]\nmode = "sometimes"',
    '[hosts.h]\njump = "missing"',
    '[hosts.h]\nfrobnicate = 1',
    '[hosts.h]\nkind = "docker"',
    '[hosts.h]\npath_map = [ { local = "/a" } ]',
])
def test_errors(bad):
    with pytest.raises(C.ConfigError):
        parse(bad)


def test_cli_init_config_never_overwrites(tmp_path):
    import subprocess
    from pathlib import Path
    target = tmp_path / "hosts.toml"
    env = dict(os.environ, HOME=str(tmp_path), TASK_SPOOLER_CONFIG=str(target),
               PYTHONPATH=str(Path(__file__).resolve().parent.parent))
    run = lambda: subprocess.run([sys.executable, "-m", "gpu_spooler.task", "--init-config"],
                                 env=env, capture_output=True, text=True)
    r = run()
    assert r.returncode == 0 and target.exists()
    assert C.load_config(target).mode == "auto"
    target.write_text("# mine\n")
    r = run()
    assert r.returncode != 0 and target.read_text() == "# mine\n"


def test_cli_list_with_builtin_config(tmp_path):
    import subprocess
    from pathlib import Path
    env = dict(os.environ, HOME=str(tmp_path), TASK_SPOOLER_CONFIG=str(tmp_path / "none.toml"),
               PYTHONPATH=str(Path(__file__).resolve().parent.parent))
    r = subprocess.run([sys.executable, "-m", "gpu_spooler.task", "-l"], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0 and "No tasks." in r.stdout
