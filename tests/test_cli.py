"""The command line: subcommand parsing."""
import argparse

import pytest

from gpu_spooler.gpu_status import add_arguments
from gpu_spooler.task import _strip_dashdash, build_parser, main


def parse(*argv):
    return build_parser().parse_args(list(argv))


def test_submit_defaults_and_command():
    a = parse("submit", "python3", "train.py", "--epochs", "3")
    assert a.gpus == 1 and a.hosts is None and a.command == ["python3", "train.py", "--epochs", "3"]
    a = parse("submit", "-g", "0", "-H", "cuda07", "-t", "RTX A5000", "-m", "40", "-n", "x",
              "-C", "/d", "-q", "--", "./run.sh", "-v")
    assert (a.gpus, a.hosts, a.gpu_type, a.min_mem, a.name, a.dir, a.quiet) == (
        0, "cuda07", "RTX A5000", 40, "x", "/d", True)
    assert _strip_dashdash(a.command) == ["./run.sh", "-v"]


def test_short_flags_mean_the_same_everywhere():
    """-H is hosts and -C the directory wherever they exist."""
    p = build_parser()
    sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
    for name, c in sub.choices.items():
        for act in c._actions:
            for flag, dest in (("-H", "hosts"), ("-C", "dir"), ("-q", "quiet"), ("-f", None)):
                if flag in act.option_strings and dest:
                    assert act.dest == dest, (name, flag)
            if "-f" in act.option_strings:
                assert act.dest in ("follow", "free"), name    # follow / gpus --free


def test_other_commands_parse():
    assert parse("ls", "-a").all
    assert parse("logs", "-f", "3").follow and parse("logs", "3").id == 3
    assert parse("wait", "3", "4", "--timeout", "5").ids == [3, 4]
    assert parse("kill", "1", "2").ids == [1, 2] and parse("kill", "--all").all
    assert parse("sync", "--check", "-H", "ecs").check
    f = parse("fetch", "results", "logs", "--after", "3", "--dry-run")
    assert f.paths == ["results", "logs"] and f.after == [3] and f.dry_run
    assert parse("daemon", "status").action == "status"
    with pytest.raises(SystemExit):
        parse("daemon", "restart")


def test_kill_needs_ids_or_all(capsys):
    with pytest.raises(SystemExit, match="IDs or --all"):
        main(["kill"])
    with pytest.raises(SystemExit, match="IDs or --all"):
        main(["kill", "3", "--all"])


def test_no_arguments_prints_help(capsys):
    main([])
    out = capsys.readouterr().out
    for c in ("submit", "list", "logs", "wait", "kill", "sync", "fetch", "gpus", "exec",
              "daemon", "config", "tui"):
        assert f"task {c}" in out


def test_gpu_status_flags():
    p = argparse.ArgumentParser()
    add_arguments(p)
    a = p.parse_args([])
    assert (a.hosts, a.workers, a.free, a.detail) == (None, 16, False, False)
    a = p.parse_args(["-H", "ecs", "-f", "-d", "--workers", "4"])
    assert (a.hosts, a.workers, a.free, a.detail) == ("ecs", 4, True, True)
    with pytest.raises(SystemExit):
        p.parse_args(["--server", "ecs"])
