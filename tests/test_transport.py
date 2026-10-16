import os
import sys
from pathlib import Path

import pytest

from gpu_spooler import config as C
from gpu_spooler.transport import Transport

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

GOLDEN = Path(__file__).parent / "golden" / "ssh_config.builtin"


def test_builtin_ssh_config_matches_golden(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))       # no ~/.ssh/config to Include
    t = Transport(C.builtin_config())
    text = t.render_ssh_config("direct").replace(str(t.cm_dir), "{CM_DIR}")
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(text)
    assert text == GOLDEN.read_text()
    assert t.mode() == "direct"
    assert "ProxyJump" not in text


def test_jump_mode_chains_each_hop(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = C.parse_config(tomllib.loads(C.config_template("u")))
    t = Transport(cfg)
    text = t.render_ssh_config("jump")
    blocks = {b.split("\n", 1)[0]: b for b in text.split("\n\n") if b.startswith("Host ")}
    assert "ProxyJump" not in blocks["Host ts-entry"]
    assert "ProxyJump ts-entry" in blocks["Host ts-barretts"]
    assert "ControlPersist 12h" in blocks["Host ts-barretts"]
    assert "ProxyJump ts-barretts" in blocks["Host cuda14"]
    assert "HostName cuda14.ecs.vuw.ac.nz" in blocks["Host cuda14"]
    assert "," not in blocks["Host cuda14"]
    direct = t.render_ssh_config("direct")
    assert "ProxyJump" not in direct and "ts-entry" not in direct


def test_include_user_ssh_config_last(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "config").write_text("Host x\n")
    text = Transport(C.builtin_config()).render_ssh_config("direct")
    assert text.rstrip().splitlines()[-1].strip() == f"Include {tmp_path}/.ssh/config"


def test_auto_mode_probe(tmp_path, monkeypatch):
    import socket
    from gpu_spooler import transport as T
    monkeypatch.setattr(T, "NETWORK_CACHE", tmp_path / "network.json")
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    port = srv.getsockname()[1]
    cfg = C.parse_config(tomllib.loads(f'''
[network]
probe = "127.0.0.1:{port}"
[jumps.j]
chain = ["gw.invalid"]
[hosts.w]
jump = "j"
'''))
    assert Transport(cfg).mode() == "direct"
    srv.close()
    assert Transport(cfg).mode() == "direct"            # cached
    assert Transport(cfg).mode(refresh=True) == "jump"   # port now closed
    assert "jump" in Transport(cfg).describe_mode(probe=False)


@pytest.mark.integration
def test_fake_cluster_control_sockets_for_each_hop(ts, fake_cluster):
    logs = {h: fake_cluster / "run" / f"sshd_{h}.log" for h in ("entry", "barretts", "cuda01")}
    before = {h: p.read_text().count("Accepted publickey") for h, p in logs.items()}
    r = ts.run("-R", "echo hi-$FAKE_HOST")
    assert "hi-cuda01" in r.stdout and "hi-cuda02" in r.stdout
    socks = os.listdir(ts.state / "cm")
    assert len(socks) == 4          # entry, barretts, cuda01, cuda02
    for h, p in logs.items():       # traffic really went through every hop
        assert p.read_text().count("Accepted publickey") > before[h], h
    # Masters are reused: a second fan-out makes no new logins.
    mid = {h: p.read_text().count("Accepted publickey") for h, p in logs.items()}
    ts.run("-R", "true")
    assert {h: p.read_text().count("Accepted publickey") for h, p in logs.items()} == mid


@pytest.mark.integration
def test_hosts_and_gpu_status_render(ts):
    fake_out = ts.run("--hosts").stdout
    assert "cuda01" in fake_out and "cuda02" in fake_out and "local" in fake_out
    assert "network: jump (forced)" in fake_out
    assert "unreachable" not in fake_out
    import subprocess
    r = subprocess.run([sys.executable, "-m", "gpu_spooler.gpu_status", "--server", "fake"],
                       env=ts.env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "A40" in r.stdout and "RTX A5000" in r.stdout
    assert "0 servers unreachable" in r.stdout.replace("\x1b[92m", "").replace("\x1b[0m", "")
