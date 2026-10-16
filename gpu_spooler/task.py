#!/usr/bin/env python3
"""
task - GPU task spooler for SSH-reachable GPU servers (VUW ECS by default)

The background daemon auto-starts on first job submission and dispatches
queued tasks to GPU servers as GPUs become free. Jobs run detached on the
server, so they survive the SSH connection (or this machine) going away.

Submission:
  task -G 1 python3 train.py             queue a job needing 1 GPU
  task -G 2 -n "experiment A" ./run.sh   queue with a label
  task -G 1 -H cuda14,cuda15 ./run.sh    only run on these hosts / groups / tags
  task -G 0 ./preprocess.sh              CPU-only job (hosts with cpu_slots)

Inspection:
  task -l  / --list                      list tasks (most recent 200)
  task -L  / --all                       list all tasks (no limit)
  task -i 3 / --info 3                   detailed info for task 3
  task -o 3 / --output 3                 print saved output of task 3
  task -f 3 / --follow 3                 stream live output of task 3

Management:
  task -k 3 / --kill 3                   kill / cancel task 3
  task -K   / --kill-all                 kill all running & queued tasks, stop daemon
  task -c   / --clear                    remove finished tasks from the list
  task -w 3 / --wait 3                   block until task 3 finishes
  task -r 3 / --resubmit 3               queue a copy of task 3

Git repos (when submitting from a machine without the servers' filesystem):
  task --sync                            push this repo's working tree to the servers
  task --repo-status                     is the servers' copy up to date?

For an interactive view of the queue, logs and GPUs, run `tui`.

Hosts / daemon:
  task --gpu-status                      show GPU availability across all servers
  task --hosts                           show configured hosts and network mode
  task --connect                         log in to jump hosts once (password/OTP)
  task --ssh cuda14                      interactive shell on a host
  task --init-config                     write ~/.task-spooler/hosts.toml template
  task --stop                            stop the background daemon
  task --daemon                          (re)start the daemon explicitly
"""

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from gpu_spooler import core
from gpu_spooler.config import ConfigError, config_path, config_template
from gpu_spooler.core import fmt_duration, fmt_ts, get_cfg, get_transport, truncate
from gpu_spooler.gpu import query_hosts
from gpu_spooler.transport import auth_hint

# ── CLI helpers ───────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
ORANGE = "\033[33m"

STATUS_COLOR = {
    "queued":    DIM,
    "running":   GREEN,
    "done":      CYAN,
    "failed":    RED,
    "killed":    YELLOW,
    "cancelled": YELLOW,
    "lost":      ORANGE,
}


def _col(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"

# ── Commands ──────────────────────────────────────────────────────────────────

def _print_submitted(sub: core.Submitted) -> None:
    label = f'  "{sub.name}"' if sub.name else ""
    print(f"Queued task {sub.task_id}{label}: {sub.command}")
    need = "CPU only" if sub.num_gpus == 0 else f"{sub.num_gpus} GPU(s) (min {sub.min_mem}GB VRAM)"
    print(f"  Requires: {need}   Working dir: {sub.working_dir}")
    if sub.host_spec:
        print(f"  Hosts: {', '.join(h.name for h in sub.candidates)}")
    for w in sub.warnings:
        print(_col(f"  Warning: {w}", YELLOW))
    daemon_status = (
        _col("running", GREEN) if core.daemon_is_running()
        else _col("NOT running — start with: task --daemon", RED)
    )
    print(f"  Daemon: {daemon_status}")


def cmd_submit(args) -> None:
    _print_submitted(core.submit(
        core.command_from_argv(args.command), args.gpus, args.min_mem, args.name,
        args.host, args.dir))


def cmd_resubmit(args) -> None:
    _print_submitted(core.resubmit(args.resubmit))


def cmd_list(args) -> None:
    con = core.db_open()
    rows = core.list_tasks(None if getattr(args, "all", False) else 200, con)
    if not rows:
        print("No tasks.")
        return

    hdr = (
        f"{'ID':>4}  {'STATUS':10}  {'G':>1}  {'HOST':<10}  {'SERVER':<14}  "
        f"{'GPU':>4}  {'SUBMITTED':11}  {'RUNTIME':>8}  LABEL / COMMAND"
    )
    sep = "─" * max(len(hdr), 80)
    print(f"\n{BOLD}{hdr}{RESET}")
    print(sep)

    for r in reversed(rows):
        status  = r["status"]
        color   = STATUS_COLOR.get(status, "")
        label   = truncate(core.one_line(r["name"] or r["command"]), 42)
        spec    = truncate(r["host_spec"] or "-", 10)
        server  = r["server"] or "-"
        gpus    = r["gpu_indices"] or "-"

        print(
            f"{r['id']:>4}  {_col(f'{status:10}', color)}  {r['num_gpus']:>1}  "
            f"{spec:<10}  {server:<14}  {gpus:>4}  {fmt_ts(r['submitted_at']):11}  "
            f"{fmt_duration(core.runtime(r)):>8}  {label}"
        )

    print(sep)
    shown = len(rows)
    counts = core.task_counts(con)
    daemon_str = (
        _col("running", GREEN) if core.daemon_is_running()
        else _col("stopped", RED)
    )
    total = counts["total"]
    truncated = f"  {_col(f'(showing {shown} of {total})', DIM)}" if shown < total else ""
    print(
        f"\n{total} task(s){truncated}  "
        f"{_col(str(counts['running']), GREEN)} running  "
        f"{_col(str(counts['queued']), YELLOW)} queued  "
        f"daemon: {daemon_str}  "
        f"network: {get_transport().describe_mode(probe=False)}\n"
    )


def cmd_info(args) -> None:
    r = core.get_task(args.id)
    print(f"\n{BOLD}Task {r['id']}{RESET}")
    for label, value in core.task_details(r):
        if label == "Status":
            value = _col(value, STATUS_COLOR.get(value, ""))
        print(f"  {label + ':':<13}{value}")
    print()


def cmd_output(args) -> None:
    con = core.db_open()
    r = con.execute("SELECT log_file, status FROM tasks WHERE id=?", (args.id,)).fetchone()
    if not r:
        print(f"Task {args.id} not found.")
        sys.exit(1)
    if not r["log_file"] or not Path(r["log_file"]).exists():
        print(f"No output log for task {args.id} yet.")
        sys.exit(1)
    print(Path(r["log_file"]).read_text())


def cmd_follow(args) -> None:
    con = core.db_open()
    r = con.execute("SELECT log_file, status FROM tasks WHERE id=?", (args.id,)).fetchone()
    if not r:
        print(f"Task {args.id} not found.")
        sys.exit(1)

    log = r["log_file"]
    status = r["status"]

    # Wait for the log file to appear if task is queued/starting
    if not log or not Path(log).exists():
        if status in ("queued", "running"):
            print(f"Task {args.id} has not started yet. Waiting...", flush=True)
            for _ in range(60):   # wait up to 60 s
                time.sleep(1)
                r = con.execute(
                    "SELECT log_file, status FROM tasks WHERE id=?", (args.id,)
                ).fetchone()
                log = r["log_file"]
                if log and Path(log).exists():
                    break
            else:
                print("Timed out waiting for task to start.")
                sys.exit(1)
        else:
            print(f"No output log for task {args.id}.")
            sys.exit(1)

    # tail -f equivalent
    print(f"Following task {args.id} (Ctrl+C to stop)...\n")
    try:
        subprocess.run(["tail", "-n", "+1", "-f", log])
    except KeyboardInterrupt:
        pass


def cmd_kill(args) -> None:
    res = core.kill(args.id)
    if res.action == "killed":
        print(res.message if res.confirmed else _col(f"Warning: {res.message}", YELLOW))
        print(f"Task {args.id} marked as killed.")
    else:
        print(res.message)


def cmd_kill_all(args) -> None:
    res = core.kill_all()
    for k in res.unconfirmed:
        print(_col(f"Task {k.task_id}: Warning: {k.message}", YELLOW))
    print(f"Cancelled {res.cancelled} queued task(s), killed {res.killed} running task(s).")
    if core.daemon_is_running():
        cmd_stop(args)
    core.reset_history()
    print("Task history cleared. Next task will be ID 1.")


def cmd_clear(args) -> None:
    res = core.clear_finished()
    print(f"Removed {res.tasks} finished task(s) and {res.log_files} log file(s)."
          + (_col(f"  Could not clean job directories on: {', '.join(res.failed_hosts)}", DIM)
             if res.failed_hosts else ""))


def cmd_sync(args) -> None:
    info = core.repo_for(args.dir, args.host)
    print(f"Syncing {BOLD}{info.repo.name}{RESET} ({info.repo.describe()}) to "
          f"{len(info.copies)} location(s)  {DIM}network: "
          f"{get_transport().describe_mode()}{RESET}")
    failed = 0
    for res in core.sync_repo(info):
        status = _col("OK  ", GREEN) if res.ok else _col("FAIL", RED)
        failed += not res.ok
        print(f"  {res.target.label:<10} {status}  {res.message}")
    busy = core.running_in([t.path for t in info.copies])
    if busy:
        print(_col(f"  Note: {busy} running task(s) use this checkout and now see the new "
                   "files.", YELLOW))
    if failed:
        sys.exit(1)


def cmd_repo_status(args) -> None:
    info = core.repo_for(args.dir, args.host)
    print(f"{BOLD}{info.repo.name}{RESET}  local: {info.repo.root}  at {info.repo.describe()}")
    stale = 0
    for target, problem in core.repo_status(info.repo, info.copies):
        stale += problem is not None
        status = _col("up to date", GREEN) if problem is None else _col(problem, YELLOW)
        print(f"  {target.label:<10} {target.path}  {status}")
        print(f"  {'':<10} {DIM}used by: {truncate(', '.join(target.members), 70)}{RESET}")
    if stale:
        sys.exit(1)


def cmd_daemon(args) -> None:
    pid = core.start_daemon()
    if pid is None:
        print("Daemon is already running.")
    else:
        print(f"Daemon started (PID {pid}). Logs: {core.DAEMON_LOG}")


def cmd_stop(args) -> None:
    pid = core.stop_daemon()
    if pid is None:
        print("Daemon is not running.")
    else:
        print(f"Daemon (PID {pid}) stopped. Running jobs keep running and are "
              "re-adopted when it restarts.")


def cmd_wait(args) -> None:
    con = core.db_open()
    print(f"Waiting for task {args.id} to finish (Ctrl+C to stop waiting)...")
    try:
        while True:
            r = con.execute("SELECT status FROM tasks WHERE id=?", (args.id,)).fetchone()
            if not r:
                print(f"Task {args.id} not found.")
                sys.exit(1)
            if r["status"] not in ("queued", "running"):
                print(f"Task {args.id} finished: {r['status']}")
                return
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopped waiting (task still running).")


def _run_on_server(host, command: str) -> tuple[bool, str]:
    """Run command on one host in a login shell; return (success, combined output)."""
    inner = "source ~/.bashrc 2>/dev/null || true; " + command
    rc, out, err = get_transport().run(host, f"exec bash -l -c {shlex.quote(inner)} < /dev/null\n",
                                 timeout=300)
    if rc == 124:
        return False, "timed out after 300s"
    output = (out + err).strip()
    hint = auth_hint(err) if rc == 255 else None
    return rc == 0, f"{output}\n({hint})" if hint else output


def cmd_run_all(args) -> None:
    command = args.run_all
    hosts = get_cfg().workers()
    n = len(hosts)
    print(f"{BOLD}Running on {n} servers:{RESET} {command}  "
          f"{DIM}network: {get_transport().describe_mode()}{RESET}\n")

    results: dict[str, tuple[bool, str]] = {}
    lock = threading.Lock()

    def one(host):
        ok, output = _run_on_server(host, command)
        with lock:
            results[host.name] = (ok, output)
            status = _col("OK  ", GREEN) if ok else _col("FAIL", RED)
            # Show the last non-empty line as a quick summary
            last = next((l for l in reversed(output.splitlines()) if l.strip()), "")
            print(f"  [{len(results):>2}/{n}] {host.name:<16} {status}  {truncate(last, 60)}",
                  flush=True)

    get_transport().fan_out(hosts, one)

    ok_count   = sum(1 for ok, _ in results.values() if ok)
    fail_count = n - ok_count

    # Print full output for failed servers
    failures = [(s, out) for s, (ok, out) in results.items() if not ok]
    if failures:
        print(f"\n{_col(f'{fail_count} server(s) failed — full output:', RED)}")
        for server, out in failures:
            print(f"\n{BOLD}── {server} ──{RESET}")
            print(out or "(no output)")

    print(
        f"\n{_col(str(ok_count), GREEN)}/{n} servers succeeded"
        + (f"  {_col(str(fail_count) + ' failed', RED)}" if fail_count else "")
        + "\n"
    )


def cmd_hosts(args) -> None:
    hosts = [h for h in get_cfg().hosts.values() if h.role == "worker"]
    mode = get_transport().mode()
    print(f"{BOLD}Querying {len(hosts)} host(s)...{RESET}", flush=True)
    results = query_hosts(get_transport(), [h for h in hosts if h.enabled])

    hdr = (f"{'HOST':<14} {'KIND':<5} {'GROUP':<8} {'VIA':<16} {'STATUS':<11} "
           f"{'GPUS':>4} {'CPU':>3}  PATH MAP")
    print(f"\n{BOLD}{hdr}{RESET}")
    print("─" * max(len(hdr), 80))
    for h in hosts:
        chain = get_cfg().chain(h)
        via = "local" if h.is_local else (
            "→".join(x.alias for x in chain) if (chain and mode == "jump") else "direct")
        r = results.get(h.name)
        if not h.enabled:
            status = _col(f"{'disabled':<11}", DIM)
        elif r and r.reachable:
            status = _col(f"{'ok':<11}", GREEN)
        else:
            status = _col(f"{'unreachable':<11}", RED)
        gpus = str(r.total_gpus) if r and r.reachable else "-"
        pmap = ", ".join(f"{l} → {rem}" for l, rem in h.path_map) or "-"
        print(f"{h.name:<14} {h.kind:<5} {h.group or '-':<8} {truncate(via, 16):<16} "
              f"{status} {gpus:>4} {h.cpu_slots or '-':>3}  {pmap}")
        if r and not r.reachable and r.error:
            print(f"{'':14} {_col(truncate(r.error, 90), DIM)}")
    print("─" * max(len(hdr), 80))
    print(f"network: {get_transport().describe_mode()}   config: {get_cfg().source_label}\n")


def _lookup_host(name: str):
    try:
        return get_cfg().get(name)
    except ConfigError as e:
        sys.exit(f"task: {e}")


def cmd_ssh(args) -> None:
    host = _lookup_host(args.ssh)
    argv = get_transport().interactive_argv(host)
    os.execvp(argv[0], argv)


def cmd_connect(args) -> None:
    mode = get_transport().mode(refresh=True)
    if mode != "jump":
        print(f"Network mode is {get_transport().describe_mode()}; no jump hosts to connect.")
        return
    failures = get_transport().connect(get_cfg().workers())
    if failures:
        sys.exit(1)
    print("Jump connections are open; the daemon will reuse them.")


def cmd_init_config(args) -> None:
    path = config_path()
    if path.exists():
        print(f"{path} already exists; not overwriting.")
        sys.exit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config_template())
    print(f"Wrote {path}. Edit it, then check with: task --hosts")


def cmd_gpu_status(args) -> None:
    from gpu_spooler.gpu_status import main as _gpu_status_main
    _gpu_status_main([])

# ── Argument parsing ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="task",
        description="GPU task spooler for SSH-reachable GPU servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        add_help=True,
    )

    # ── Submission ────────────────────────────────────────────────────────────
    p.add_argument(
        "-G", "--gpus", metavar="N", type=int,
        help="Number of GPUs required (submit mode); 0 for a CPU-only job",
    )
    p.add_argument(
        "-H", "--host", metavar="SPEC",
        help="Only run on these hosts, groups or tags (comma separated)",
    )
    p.add_argument(
        "-m", "--min-mem", metavar="GB", type=int, default=0,
        help="Minimum VRAM per GPU in GB (e.g. 40 for A40/A6000)",
    )
    p.add_argument(
        "-n", "--name", metavar="LABEL",
        help="Optional human-readable label for the task",
    )
    p.add_argument(
        "-d", "--dir", metavar="PATH",
        help="Working directory (default: current directory; translated by path_map or remote_root)",
    )

    # ── Inspection ────────────────────────────────────────────────────────────
    p.add_argument("-l", "--list",   action="store_true", help="List tasks (most recent 200)")
    p.add_argument("-L", "--all",    action="store_true", help="List all tasks (no limit)")
    p.add_argument("-i", "--info",   metavar="ID", type=int, help="Detailed info for a task")
    p.add_argument("-o", "--output", metavar="ID", type=int, help="Print saved output of a task")
    p.add_argument("-f", "--follow", metavar="ID", type=int, help="Stream live output of a task")

    # ── Management ───────────────────────────────────────────────────────────
    p.add_argument("-k",  "--kill",     metavar="ID", type=int, help="Kill or cancel a task")
    p.add_argument("-K", "--kill-all", action="store_true",    help="Kill all running and queued tasks and stop the daemon")
    p.add_argument("-c", "--clear", action="store_true", help="Remove finished tasks from the list")
    p.add_argument("-w", "--wait",  metavar="ID", type=int, help="Block until a task finishes")
    p.add_argument("-r", "--resubmit", metavar="ID", type=int,
                   help="Queue a copy of a task (same command, GPUs, hosts and directory)")

    # ── GPU / daemon ──────────────────────────────────────────────────────────
    p.add_argument("-R", "--run-all", metavar="CMD",
                   help="Run a shell command on every GPU server in parallel and report results")
    p.add_argument("--gpu-status", action="store_true", help="Show GPU availability across servers")
    p.add_argument("--daemon",     action="store_true", help="Start daemon (if not already running)")
    p.add_argument("--stop",       action="store_true", help="Stop the background daemon")
    p.add_argument("--sync",       action="store_true",
                   help="Push this git repo's working tree to its remote copies (remote_root)")
    p.add_argument("--repo-status", action="store_true",
                   help="Check whether the remote copies of this git repo are up to date")
    p.add_argument("--hosts",      action="store_true",
                   help="Show configured hosts, reachability and network mode")
    p.add_argument("--connect",    action="store_true",
                   help="Open jump-host connections interactively (password/OTP once)")
    p.add_argument("--ssh",        metavar="HOST",
                   help="Open an interactive shell on a host via the generated ssh config")
    p.add_argument("--init-config", action="store_true",
                   help="Write a commented hosts.toml template (never overwrites)")

    # Internal: daemon loop (not for user use)
    p.add_argument("--_daemon-loop", action="store_true", help=argparse.SUPPRESS)

    # Positional: the command to run (used with -G)
    p.add_argument("command", nargs=argparse.REMAINDER, help="Command to queue (used with -G)")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Internal: run the actual daemon loop
    if args._daemon_loop:
        from gpu_spooler.daemon import main as daemon_main
        daemon_main()
        return

    try:
        dispatch(parser, args)
    except core.TaskError as e:
        sys.exit(f"task: {e}")
    except ConfigError as e:
        sys.exit(f"task: config error: {e}")


def dispatch(parser, args) -> None:

    # Route to the appropriate command
    if args.gpus is not None:
        if not args.command:
            parser.error("-G requires a command, e.g.: task -G 1 python3 train.py")
        cmd_submit(args)
    elif args.list or args.all:
        cmd_list(args)
    elif args.info is not None:
        args.id = args.info;   cmd_info(args)
    elif args.output is not None:
        args.id = args.output; cmd_output(args)
    elif args.follow is not None:
        args.id = args.follow; cmd_follow(args)
    elif args.kill is not None:
        args.id = args.kill;   cmd_kill(args)
    elif args.kill_all:
        cmd_kill_all(args)
    elif args.clear:
        cmd_clear(args)
    elif args.wait is not None:
        args.id = args.wait;   cmd_wait(args)
    elif args.run_all:
        cmd_run_all(args)
    elif args.gpu_status:
        cmd_gpu_status(args)
    elif args.daemon:
        cmd_daemon(args)
    elif args.stop:
        cmd_stop(args)
    elif args.resubmit is not None:
        cmd_resubmit(args)
    elif args.sync:
        cmd_sync(args)
    elif args.repo_status:
        cmd_repo_status(args)
    elif args.init_config:
        cmd_init_config(args)
    elif args.hosts:
        cmd_hosts(args)
    elif args.ssh:
        cmd_ssh(args)
    elif args.connect:
        cmd_connect(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
