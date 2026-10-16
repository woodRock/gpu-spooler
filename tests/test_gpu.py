from gpu_spooler.gpu import parse_output

RAW = """\
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


def test_parse_output():
    gpus = parse_output(RAW)
    assert [g.index for g in gpus] == [0, 1, 2, 3]
    assert [g.is_free for g in gpus] == [True, False, False, False]
    g2 = gpus[2]
    assert [(p.pid, p.user, p.mem_used_mb, p.command) for p in g2.processes] == [
        (4242, "alice", 17990, "python")]
    assert [(p.pid, p.user) for p in gpus[3].processes] == [(5151, "bob")]
    assert round(gpus[0].mem_total_gb, 1) == 48.3


def test_parse_old_pmon_format_and_crlf():
    raw = RAW.replace("\n", "\r\n").replace(
        "    2       4242      C     99     40      -      -      -      -  17990   python",
        "    2       4242     C    99    40     -     -  17990   python")
    gpus = parse_output(raw)
    assert gpus[2].processes[0].mem_used_mb == 17990


def test_parse_empty():
    assert parse_output("") == []
