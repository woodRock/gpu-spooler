#!/usr/bin/env python3
"""
task - GPU task spooler for SSH-reachable GPU servers (VUW ECS by default)

The background daemon auto-starts on first submission and dispatches queued
tasks to GPU servers as GPUs become free. Jobs run detached on the server, so
they survive the SSH connection (or this machine) going away.

Tasks:
  task submit [-g N] [OPTIONS] -- CMD     queue CMD (1 GPU unless -g says otherwise)
  task list [-a]                          list tasks (the 200 most recent; -a: all)
  task info ID                            details of a task
  task logs [-f] ID                       print a task's output (-f: follow it)
  task wait ID... [--timeout S]           wait; the exit code says how they ended
  task kill ID... | --all                 cancel queued / kill running tasks
  task resubmit ID                        queue a copy of a task
  task clear [--reset]                    remove finished tasks (--reset: all, IDs from 1)

Repos (submitting from a laptop):
  task sync [--check]                     push this repo to the servers (--check: is it current?)
  task fetch [OPTIONS] PATH...            copy outputs back (PATH or REMOTE:LOCAL)

Cluster:
  task gpus [-d] [-f]                     GPU availability (same as gpu-status)
  task hosts                              configured hosts and network mode
  task exec [-H SPEC] -- CMD              run CMD on every (selected) server
  task ssh HOST                           interactive shell on a host
  task connect                            log in to jump hosts once (password/OTP)
  task daemon start|stop|status           the background daemon
  task config init|path                   the hosts.toml file
  task tui                                interactive view of the queue, logs and GPUs

The same short flag means the same thing everywhere: -H hosts/groups/tags,
-C directory (as with git -C), -g GPUs, -t GPU model, -n name, -q quiet,
-f follow. `task COMMAND -h` shows a command's options.

Examples:
  task submit -- python3 train.py --epochs 50
  task submit -g 2 -n "ablation" -t "RTX A5000" -- ./run.sh
  id=$(task submit -q -- python3 eval.py) && task wait "$id" && task fetch results
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
from gpu_spooler.idle import fmt_idle
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
    if sub.gpu_type:
        print(f"  GPU type: {sub.gpu_type}")
    for w in sub.warnings:
        print(_col(f"  Warning: {w}", YELLOW))
    daemon_status = (
        _col("running", GREEN) if core.daemon_is_running()
        else _col("NOT running — start with: task daemon start", RED)
    )
    print(f"  Daemon: {daemon_status}")


def _report_submitted(sub: core.Submitted, quiet: bool) -> None:
    if not quiet:
        _print_submitted(sub)
        return
    print(sub.task_id)
    for w in sub.warnings:
        print(f"task: warning: {w}", file=sys.stderr)


def cmd_submit(args) -> None:
    if not args.command:
        sys.exit("task submit: no command given, e.g.: task submit -- python3 train.py")
    _report_submitted(core.submit(
        core.command_from_argv(args.command), args.gpus, args.min_mem, args.name,
        args.hosts, args.dir, gpu_type=args.gpu_type, no_record=args.no_record), args.quiet)


def cmd_resubmit(args) -> None:
    _report_submitted(core.resubmit(args.id), args.quiet)


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

    idle = idle_tasks(rows)
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
            + (_col(f"  ⚠ {idle[r['id']]}", YELLOW) if r["id"] in idle else "")
        )

    print(sep)
    if idle:
        print(_col(f"{len(idle)} running task(s) have held their GPUs under 5% utilisation for "
                   f"{get_cfg().idle_minutes}m+ (gpu-detective). They still count against max_gpus.",
                   YELLOW))
    shown = len(rows)
    counts = core.task_counts(con)
    daemon_str = (
        _col("running", GREEN) if core.daemon_is_running()
        else _col("stopped", RED)
    )
    total = counts["total"]
    truncated = f"  {_col(f'(showing {shown} of {total})', DIM)}" if shown < total else ""
    cap = get_cfg().max_gpus
    in_use = sum(r["num_gpus"] or 0 for r in con.execute(
        "SELECT num_gpus FROM tasks WHERE status='running'"))
    print(
        f"\n{total} task(s){truncated}  "
        f"{_col(str(counts['running']), GREEN)} running  "
        f"{_col(str(counts['queued']), YELLOW)} queued  "
        + (f"GPUs {in_use}/{cap} (max_gpus)  " if cap is not None else "") +
        f"daemon: {daemon_str}  "
        f"network: {get_transport().describe_mode(probe=False)}\n"
    )


def idle_tasks(rows) -> dict[int, str]:
    """Running tasks whose every GPU gpu-detective has seen held idle: id → 'idle 1h05m'."""
    from gpu_spooler.gpu_status import load_idle
    if not any(r["status"] == "running" and r["gpu_indices"] for r in rows):
        return {}
    idle = load_idle(get_cfg())
    if not idle:
        return {}
    out = {}
    for r in rows:
        if r["status"] != "running" or not r["gpu_indices"]:
            continue
        gs = [idle.get((r["server"], int(i))) for i in r["gpu_indices"].split(",") if i.strip().isdigit()]
        if gs and all(gs):
            out[r["id"]] = fmt_idle(min(gs, key=lambda g: g.idle_seconds))
    return out


def cmd_info(args) -> None:
    r = core.get_task(args.id)
    print(f"\n{BOLD}Task {r['id']}{RESET}")
    for label, value in core.task_details(r):
        if label == "Status":
            value = _col(value, STATUS_COLOR.get(value, ""))
        print(f"  {label + ':':<13}{value}")
    print()


def cmd_logs(args) -> None:
    con = core.db_open()
    r = con.execute("SELECT log_file, status FROM tasks WHERE id=?", (args.id,)).fetchone()
    if not r:
        sys.exit(f"task: task {args.id} not found")
    log = r["log_file"]
    if not args.follow:
        if not log or not Path(log).exists():
            sys.exit(f"task: no output for task {args.id} yet ({r['status']})")
        print(Path(log).read_text(), end="")
        return

    # Wait for the log file to appear if the task is queued or starting
    if not log or not Path(log).exists():
        if r["status"] not in ("queued", "running"):
            sys.exit(f"task: no output for task {args.id} ({r['status']})")
        print(f"Task {args.id} has not started yet. Waiting...", file=sys.stderr, flush=True)
        for _ in range(60):   # wait up to 60 s
            time.sleep(1)
            log = con.execute("SELECT log_file FROM tasks WHERE id=?",
                              (args.id,)).fetchone()["log_file"]
            if log and Path(log).exists():
                break
        else:
            sys.exit("task: timed out waiting for the task to start")
    print(f"Following task {args.id} (Ctrl+C to stop)...\n", file=sys.stderr)
    try:
        subprocess.run(["tail", "-n", "+1", "-f", log])
    except KeyboardInterrupt:
        pass


def cmd_kill(args) -> None:
    if args.all == bool(args.ids):
        sys.exit("task kill: give task IDs or --all (not both)")
    if args.all:
        res = core.kill_all()
        for k in res.unconfirmed:
            print(_col(f"Task {k.task_id}: Warning: {k.message}", YELLOW))
        print(f"Cancelled {res.cancelled} queued task(s), killed {res.killed} running task(s).")
        return
    for tid in dict.fromkeys(args.ids):
        res = core.kill(tid)
        if res.action == "killed":
            print(res.message if res.confirmed else _col(f"Warning: {res.message}", YELLOW))
            print(f"Task {tid} marked as killed.")
        else:
            print(res.message)


def cmd_clear(args) -> None:
    if args.reset:
        counts = core.task_counts()
        if counts["running"] or counts["queued"]:
            sys.exit(f"task clear --reset: {counts['running']} running and {counts['queued']} "
                     "queued task(s); kill them first (task kill --all)")
    res = core.clear_finished()
    print(f"Removed {res.tasks} finished task(s) and {res.log_files} log file(s)."
          + (_col(f"  Could not clean job directories on: {', '.join(res.failed_hosts)}", DIM)
             if res.failed_hosts else ""))
    if args.reset:
        core.reset_history()
        print("Task history cleared. The next task will be ID 1.")


def cmd_sync(args) -> None:
    if args.check:
        cmd_repo_status(args)
        return
    info = core.repo_for(args.dir, args.hosts)
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


def cmd_fetch(args) -> None:
    from gpu_spooler import fetch
    done: dict = {}
    if args.after:
        done, _ = core.wait_for(list(dict.fromkeys(args.after)))
        bad = [core.describe_end(r) for r in done.values() if r["status"] != "done"]
        if bad:
            sys.exit("task: not fetching, a task did not finish cleanly: " + "; ".join(bad))
    cfg, transport = get_cfg(), get_transport()
    cwd = os.path.abspath(os.path.expanduser(args.dir)) if args.dir else core.logical_cwd()
    if core.on_ecs_filesystem(cwd) and not any(fetch._is_explicit(s) for s in args.paths):
        sys.exit("task: this directory is on the servers' filesystem already; nothing to fetch")
    try:
        hosts = cfg.select(args.hosts)
        group, items = fetch.resolve(cfg, hosts, args.paths, cwd)
        busy = core.running_in([i.remote for i in items])
        if busy and not args.after:
            print(_col(f"Warning: {busy} running task(s) work under these paths; files may be "
                       "incomplete (use --after ID to wait for them).", YELLOW))
        res = fetch.fetch(cfg, transport, hosts, args.paths, cwd,
                          force=args.force, dry_run=args.dry_run)
    except fetch.FetchError as e:
        sys.exit(f"task: {e}")
    except ConfigError as e:
        sys.exit(f"task: -H {args.hosts}: {e}")
    verb = "Would fetch" if args.dry_run else "Fetched"
    kept_any = False
    for plan in res.plans:
        it = plan.item
        where = f"{res.host.name}:{it.remote} → {os.path.relpath(it.local, cwd)}"
        if plan.kind == "missing":
            print(f"  {_col('MISSING', RED)}  {where}")
            continue
        n = len(plan.to_fetch)
        parts = [f"{len(plan.new)} new", f"{len(plan.update)} updated",
                 f"{len(plan.same)} unchanged"]
        if plan.kept:
            parts.append(_col(f"{len(plan.kept)} kept (local changes)", YELLOW))
        if plan.conflict:
            parts.append(_col(f"{len(plan.conflict)} conflict (local directory)", RED))
        skipped = res.skipped.get(it.spec, [])
        if skipped:
            parts.append(_col(f"{len(skipped)} skipped (unsafe link)", YELLOW))
        print(f"  {verb} {n:>4}  {where}  ({', '.join(parts)})")
        for rel in (plan.kept + plan.conflict)[:20]:
            print(f"           {DIM}kept: {os.path.join(it.spec, rel) if rel else it.spec}{RESET}")
        for m in skipped[:20]:
            print(f"           {DIM}skipped: {m}{RESET}")
        kept_any |= bool(plan.kept or plan.conflict)
    if kept_any:
        print(_col("Local files that changed since they were fetched, or that git cannot "
                   "restore, were kept; --force replaces them.", YELLOW))
    if args.after and not args.dry_run:
        _record_outputs(cfg, done, res)
    if any(p.kind == "missing" for p in res.plans):
        sys.exit(1)


def _record_outputs(cfg, done: dict, res) -> None:
    """With --after, what this fetch brought back new or changed is those tasks' output: record it
    with taura (if the repo keeps a taura ledger; see gpu_spooler/provenance.py). With several
    tasks, each file goes to the task that wrote it (fetch.attribute)."""
    from gpu_spooler import fetch, provenance
    outputs = []
    for plan in res.plans:
        base = plan.item.local if plan.kind == "dir" else None
        outputs += [os.path.join(base, rel) if base else plan.item.local for rel in plan.new + plan.update]
    outputs = [p for p in outputs if os.path.isfile(p)]
    rows = list(done.values())
    by_task, shared = fetch.attribute(rows, outputs)
    if shared:
        ids = ", ".join(str(r["id"]) for r in rows)
        print(_col(f"Warning: {len(shared)} file(s) could not be told apart between tasks {ids} "
                   "(no command names them, and the tasks ran at the same time); recorded on each. "
                   "Fetch each task's outputs on its own to keep them apart.", YELLOW))
        for p in shared[:10]:
            print(f"           {DIM}{os.path.relpath(p)}{RESET}")
    for row in rows:
        mine = by_task[row["id"]]
        if provenance.notify("fetch", row, cfg.taura, outputs=mine, log=lambda m: print(_col(m, YELLOW))):
            print(f"  Recorded {len(mine)} output(s) of task {row['id']} as taura run {provenance.run_id(row)}.")


def cmd_repo_status(args) -> None:
    info = core.repo_for(args.dir, args.hosts)
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
    if args.action == "stop":
        cmd_stop(args)
        return
    if args.action == "status":
        pid = core.daemon_pid()
        print(f"Daemon running (PID {pid}). Logs: {core.DAEMON_LOG}" if pid
              else "Daemon is not running (it starts on the next submit).")
        return
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
    """
    Wait for tasks to finish; exit with a code reflecting how they ended
    (core.wait_exit_code). Reports go to stderr; stdout stays empty.
    """
    ids = list(dict.fromkeys(args.ids))
    try:
        done, pending = core.wait_for(ids, timeout=args.timeout)
    except KeyboardInterrupt:
        print("task: stopped waiting (tasks keep running)", file=sys.stderr)
        sys.exit(core.EXIT_KILLED)
    for tid in ids:
        if tid in done:
            print(core.describe_end(done[tid]), file=sys.stderr)
        else:
            print(f"task {tid}: still {core.get_task(tid)['status']} "
                  f"(timed out after {args.timeout:g}s)", file=sys.stderr)
    if pending:
        sys.exit(core.EXIT_TIMEOUT)
    codes = [core.wait_exit_code(done[tid]) for tid in sorted(ids)]
    sys.exit(next((c for c in codes if c), 0))


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


def cmd_exec(args) -> None:
    if not args.command:
        sys.exit("task exec: no command given, e.g.: task exec -- nvidia-smi -L")
    command = core.command_from_argv(args.command)
    try:
        hosts = get_cfg().select(args.hosts)
    except ConfigError as e:
        sys.exit(f"task exec: -H {args.hosts}: {e}")
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
    host = _lookup_host(args.host)
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


def cmd_config(args) -> None:
    path = config_path()
    if args.action == "path":
        print(path if path.exists() else f"{path} (does not exist; built-in defaults are used)")
        return
    if path.exists():
        print(f"{path} already exists; not overwriting.")
        sys.exit(1)
    import getpass
    user = args.user or getpass.getuser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config_template(user))
    how = "" if args.user else " (your local username; use --user if your ECS one differs)"
    print(f"Wrote {path} for user '{user}'{how}. Check it with: task hosts")


def cmd_gpus(args) -> None:
    from gpu_spooler.gpu_status import run as gpu_status_run
    gpu_status_run(args)


def cmd_tui(args) -> None:
    from gpu_spooler.tui import main as tui_main
    tui_main()

# ── Argument parsing ──────────────────────────────────────────────────────────

# Shown at the end of `task COMMAND -h`.
EXAMPLES = {
    "submit": [
        ("python3 train.py --epochs 50", "1 GPU on any free server"),
        ('-g 2 -n "ablation" -- ./run.sh --lr 3e-4', "2 GPUs, with a label"),
        ('-t "RTX A5000" -- python3 eval.py', "only on RTX A5000 GPUs"),
        ("-m 40 -H cuda14,cuda15 -- python3 big.py", "48 GB cards on these hosts"),
        ("-g 0 -H local -- 'make plots && echo done'", "CPU job; one argument = shell code"),
        ('id=$(task submit -q -- python3 train.py)', "keep the ID for wait / fetch"),
    ],
    "list": [("", "the 200 most recent tasks"), ("-a", "every task")],
    "info": [("12", "server, GPUs, directories, times, exit code")],
    "logs": [("12", "everything task 12 printed so far"),
             ("-f 12", "keep following it (Ctrl+C stops following, not the task)")],
    "wait": [("12", "block until task 12 ends; exit code says how"),
             ("12 13 14 --timeout 3600", "all three, for at most an hour"),
             ('12 && task fetch results', "fetch only if the task succeeded")],
    "kill": [("12", "cancel it if queued, kill it if running"),
             ("12 13 14", "several at once"), ("--all", "every queued and running task")],
    "resubmit": [("12", "queue a copy (same command, GPUs, hosts, directory)")],
    "clear": [("", "forget finished tasks and delete their logs"),
              ("--reset", "forget everything and restart IDs at 1")],
    "sync": [("", "push this repo's working tree (uncommitted changes included)"),
             ("--check", "is the servers' copy up to date? (exit 1 if not)")],
    "fetch": [("results", "this repo's results/ from the servers' copy"),
              ("results --after 12 13", "once tasks 12 and 13 are done"),
              ("results --dry-run", "show what would change"),
              ("/vol/ecrg-solar/me/run/results:out", "explicit REMOTE:LOCAL")],
    "gpus": [("", "one row per server"), ("-f", "only servers with a free GPU"),
             ("-d -H cuda14", "one row per GPU on cuda14")],
    "hosts": [("", "route, reachability and GPU count of every configured host")],
    "exec": [("-- nvidia-smi -L", "on every server"),
             ('-H cuda14,cuda15 -- "df -h /vol/ecrg-solar"', "on two servers")],
    "ssh": [("cuda14", "a shell on cuda14, through the jump hosts if needed")],
    "connect": [("", "type the jump host's password / OTP once; the daemon reuses it")],
    "daemon": [("status", "is it running?"), ("stop", "running jobs keep running"),
               ("start", "it also starts by itself on submit")],
    "config": [("init --user woodj4", "write ~/.gpu-spooler/hosts.toml (never overwrites)"),
               ("path", "which config file is in use")],
    "tui": [("", "live queue, logs and a GPU heatmap (q quits)")],
}


def _examples(name: str) -> str | None:
    rows = EXAMPLES.get(name)
    if not rows:
        return None
    cmds = [a if a.startswith("id=") else f"task {name} {a}".rstrip() for a, _ in rows]
    width = max(map(len, cmds))
    lines = [f"  {c:<{width}}  # {what}" for c, (_, what) in zip(cmds, rows)]
    return "Examples:\n" + "\n".join(lines)


class _Formatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, max_help_position=30)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="task", description=__doc__, formatter_class=_Formatter,
        usage="task COMMAND [OPTIONS] [ARGS]",
    )
    p.add_argument("-V", "--version", action="version", version=f"%(prog)s {_version()}")
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND", help=argparse.SUPPRESS)

    def cmd(name: str, func, help: str, aliases=()):
        c = sub.add_parser(name, aliases=list(aliases), help=help, description=help,
                           formatter_class=_Formatter, epilog=_examples(name))
        c.set_defaults(func=func)
        return c

    def hosts_arg(c, what="Only these hosts, groups or tags (comma separated)"):
        c.add_argument("-H", "--hosts", metavar="SPEC", help=what)

    def dir_arg(c, what):
        c.add_argument("-C", "--dir", metavar="DIR", help=what)

    def ids_arg(c, nargs="+"):
        c.add_argument("ids", metavar="ID", type=int, nargs=nargs)

    # ── tasks ────────────────────────────────────────────────────────────────
    c = cmd("submit", cmd_submit, "Queue a command to run on a GPU server")
    c.add_argument("-g", "--gpus", metavar="N", type=int, default=1,
                   help="GPUs the task needs (default 1; 0 = CPU-only, on hosts with cpu_slots)")
    hosts_arg(c, "Only run on these hosts, groups or tags (comma separated)")
    c.add_argument("-t", "--gpu-type", metavar="MODEL",
                   help="Only GPUs nvidia-smi names this model, e.g. \"RTX A5000\" (exact, "
                        "ignoring case and 'NVIDIA '; comma separated for several)")
    c.add_argument("-m", "--min-mem", metavar="GB", type=int, default=0,
                   help="Only GPUs with at least this much memory, e.g. 40")
    c.add_argument("-n", "--name", metavar="LABEL", help="A label shown in the task list")
    dir_arg(c, "Working directory (default: here; mapped by path_map or remote_root)")
    c.add_argument("-q", "--quiet", action="store_true",
                   help="Print only the task ID (for scripts)")
    c.add_argument("--no-record", action="store_true",
                   help="Don't record this task as a taura run (a one-off check, not an experiment)")
    c.add_argument("command", nargs=argparse.REMAINDER, metavar="-- CMD",
                   help="The command; one quoted argument is run as shell code")

    c = cmd("list", cmd_list, "List tasks", aliases=["ls"])
    c.add_argument("-a", "--all", action="store_true", help="All tasks, not only the last 200")

    c = cmd("info", cmd_info, "Show the details of a task")
    c.add_argument("id", metavar="ID", type=int)

    c = cmd("logs", cmd_logs, "Print a task's output")
    c.add_argument("id", metavar="ID", type=int)
    c.add_argument("-f", "--follow", action="store_true", help="Keep printing new output")

    c = cmd("wait", cmd_wait, "Wait for tasks to finish",)
    c.description = ("Wait for tasks to finish. Exit code: 0 all done, the job's exit code if "
                     "one failed, 130 killed or cancelled, 75 lost, 124 timed out.")
    ids_arg(c)
    c.add_argument("--timeout", metavar="S", type=float,
                   help="Give up after S seconds (exit 124; the tasks keep running)")

    c = cmd("kill", cmd_kill, "Cancel queued or kill running tasks")
    ids_arg(c, nargs="*")
    c.add_argument("--all", action="store_true", help="Every queued and running task")

    c = cmd("resubmit", cmd_resubmit, "Queue a copy of a task (same command and options)")
    c.add_argument("id", metavar="ID", type=int)
    c.add_argument("-q", "--quiet", action="store_true", help="Print only the new task ID")

    c = cmd("clear", cmd_clear, "Remove finished tasks, their logs and job directories")
    c.add_argument("--reset", action="store_true",
                   help="Also forget everything else and restart IDs at 1 (no task may be "
                        "queued or running)")

    # ── repos ────────────────────────────────────────────────────────────────
    c = cmd("sync", cmd_sync, "Push this git repo's working tree to its copy on the servers")
    c.add_argument("--check", action="store_true",
                   help="Only report whether the copies are up to date (exit 1 if not)")
    hosts_arg(c)
    dir_arg(c, "The repo (default: the one containing the current directory)")

    c = cmd("fetch", cmd_fetch, "Copy outputs back from the servers' copy of this repo")
    c.description = ("Copy PATHs back from the servers. A relative PATH maps into this repo's "
                     "copy (or a path_map entry); REMOTE:LOCAL names both sides. Links are "
                     "never followed and locally changed files are kept.")
    c.add_argument("paths", metavar="PATH", nargs="+")
    c.add_argument("--after", metavar="ID", type=int, nargs="+",
                   help="Wait for these tasks first; fetch only if all are done")
    c.add_argument("--force", action="store_true", help="Also replace locally changed files")
    c.add_argument("--dry-run", action="store_true", help="Show what would change, copy nothing")
    hosts_arg(c, "Fetch from this host group (needed if several are selected)")
    dir_arg(c, "Resolve PATHs from this directory instead of the current one")

    # ── cluster ──────────────────────────────────────────────────────────────
    from gpu_spooler.gpu_status import add_arguments as gpu_status_arguments
    c = cmd("gpus", cmd_gpus, "Show GPU availability across servers (same as gpu-status)")
    gpu_status_arguments(c)

    cmd("hosts", cmd_hosts, "Show configured hosts, reachability and network mode")

    c = cmd("exec", cmd_exec, "Run a shell command on every (selected) server in parallel")
    hosts_arg(c)
    c.add_argument("command", nargs=argparse.REMAINDER, metavar="-- CMD")

    c = cmd("ssh", cmd_ssh, "Open an interactive shell on a host")
    c.add_argument("host", metavar="HOST")

    cmd("connect", cmd_connect, "Log in to the jump hosts once (password / OTP)")

    c = cmd("daemon", cmd_daemon, "Start, stop or check the background daemon")
    c.add_argument("action", choices=["start", "stop", "status"])

    c = cmd("config", cmd_config, "Write a hosts.toml template, or show where it is")
    c.add_argument("action", choices=["init", "path"])
    c.add_argument("-u", "--user", metavar="NAME",
                   help="With init: your username on the servers (default: your local one)")

    cmd("tui", cmd_tui, "Interactive view of the queue, logs and GPUs")
    return p


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("gpu-spooler")
    except Exception:
        return "unknown"


def _strip_dashdash(argv: list[str]) -> list[str]:
    return argv[1:] if argv[:1] == ["--"] else argv


def run(argv: list[str]) -> None:
    """Parse one subcommand argv and run it."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return
    if hasattr(args, "command"):
        args.command = _strip_dashdash(args.command)
    args.func(args)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    try:
        run(argv)
    except core.TaskError as e:
        print(f"task: {e}", file=sys.stderr)
        sys.exit(1)
    except ConfigError as e:
        sys.exit(f"task: config error: {e}")


if __name__ == "__main__":
    main()
