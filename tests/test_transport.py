"""The ssh transport on the fake cluster: one ControlMaster per hop, reused (the generated ssh configs
are checked in crates/gpu-spooler/tests/golden.rs)."""
import os

import pytest


@pytest.mark.integration
def test_fake_cluster_control_sockets_for_each_hop(ts, fake_cluster):
    logs = {h: fake_cluster / "run" / f"sshd_{h}.log" for h in ("entry", "barretts", "cuda01")}
    before = {h: p.read_text().count("Accepted publickey") for h, p in logs.items()}
    r = ts.run("exec", "echo hi-$FAKE_HOST")
    assert "hi-cuda01" in r.stdout and "hi-cuda02" in r.stdout
    socks = os.listdir(ts.state / "cm")
    assert len(socks) == 4          # entry, barretts, cuda01, cuda02
    for h, p in logs.items():       # traffic really went through every hop
        assert p.read_text().count("Accepted publickey") > before[h], h
    # Masters are reused: a second fan-out makes no new logins.
    mid = {h: p.read_text().count("Accepted publickey") for h, p in logs.items()}
    ts.run("exec", "true")
    assert {h: p.read_text().count("Accepted publickey") for h, p in logs.items()} == mid


@pytest.mark.integration
def test_hosts_and_gpu_status_render(ts):
    fake_out = ts.run("hosts").stdout
    assert "cuda01" in fake_out and "cuda02" in fake_out and "local" in fake_out
    assert "network: jump (forced)" in fake_out
    assert "unreachable" not in fake_out
    r = ts.run("gpus", "-H", "fake")
    assert "A40" in r.stdout and "RTX A5000" in r.stdout
    assert "0 servers unreachable" in r.stdout.replace("\x1b[92m", "").replace("\x1b[0m", "")
