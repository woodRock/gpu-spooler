#!/usr/bin/env python3
"""
GPU server status checker (also `task gpus`)

Queries every configured host in parallel (see ~/.gpu-spooler/hosts.toml),
runs nvidia-smi, and displays a table showing free/busy GPUs, users, and
running processes.

Usage:
    gpu-status [-d/--detail] [-f/--free] [-H/--hosts SPEC]
"""

import argparse
import sys

from gpu_spooler.config import Config, ConfigError, load_config
from gpu_spooler.gpu import GpuProcess, ServerResult, gpu_model, gpu_type_matches, query_host
from gpu_spooler.transport import Transport

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
ORANGE = "\033[33m"


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def mb_to_gb(mb: int) -> str:
    return f"{mb / 1024:.1f}"


def bar(used_mb: int, total_mb: int, width: int = 10) -> str:
    """Simple ASCII utilisation bar."""
    if total_mb == 0:
        return " " * width
    frac = used_mb / total_mb
    filled = round(frac * width)
    bar_str = "█" * filled + "░" * (width - filled)
    color = GREEN if frac < 0.5 else (YELLOW if frac < 0.85 else RED)
    return f"{color}{bar_str}{RESET}"


def label_mismatches(results: list[ServerResult], cfg: Config) -> list[tuple[str, str, str]]:
    """(host, configured gpu_type, what nvidia-smi reports) where the two disagree."""
    out = []
    for r in sorted(results, key=lambda x: x.name):
        host = cfg.hosts.get(r.name)
        if not (host and host.gpu_type and r.reachable and r.gpus):
            continue
        if not all(gpu_type_matches(host.gpu_type, m) for m in r.models):
            out.append((r.name, host.gpu_type, ", ".join(r.models)))
    return out


def print_summary_table(results: list[ServerResult], cfg: Config, free_only: bool = False) -> None:
    """One row per server. GPU TYPE is what nvidia-smi reports (the configured gpu_type
    only for hosts that reported no GPUs); * marks a configured gpu_type that disagrees."""
    mismatched = {h for h, _, _ in label_mismatches(results, cfg)}
    header = (
        f"{'SERVER':<14} {'GPU TYPE':<24} {'GPUs':>4}  {'FREE':>4}  "
        f"{'VRAM FREE':>10}  {'VRAM TOTAL':>10}  {'USERS / PROCESSES'}"
    )
    sep = "─" * len(header)
    print(f"\n{BOLD}{header}{RESET}")
    print(sep)

    for r in sorted(results, key=lambda x: (not x.reachable, x.name)):
        if free_only and r.reachable and r.free_gpus == 0:
            continue

        host = cfg.hosts.get(r.name)
        gpu_type = "/".join(r.models) or (host and host.gpu_type) or "-"
        if r.name in mismatched:
            gpu_type += "*"

        if not r.reachable:
            print(
                f"{r.name:<14} {colorize('UNREACHABLE', RED):<33}  {r.error[:60]}"
            )
            continue

        total = r.total_gpus
        free  = r.free_gpus

        if free == total:
            free_str = colorize(str(free), GREEN)
        elif free == 0:
            free_str = colorize(str(free), RED)
        else:
            free_str = colorize(str(free), YELLOW)

        # Collect unique users and their processes
        user_procs: dict[str, set[str]] = {}
        for g in r.gpus:
            for p in g.processes:
                user_procs.setdefault(p.user, set()).add(p.command)

        who = "; ".join(
            f"{colorize(u, CYAN)}: {', '.join(sorted(cmds))}"
            for u, cmds in sorted(user_procs.items())
        ) or colorize("idle", DIM)

        vram_free  = f"{mb_to_gb(r.free_vram_mb):>5} GB"
        vram_total = f"{mb_to_gb(r.total_vram_mb):>5} GB"

        print(
            f"{r.name:<14} {gpu_type:<24} {total:>4}  {free_str:>12}  "
            f"{vram_free:>10}  {vram_total:>10}  {who}"
        )

    print(sep)
    print_label_warnings(results, cfg)


def print_label_warnings(results: list[ServerResult], cfg: Config) -> None:
    rows = label_mismatches(results, cfg)
    if not rows:
        return
    src = cfg.source_label
    print(colorize(f"* gpu_type in {src} disagrees with nvidia-smi:", YELLOW))
    for name, configured, live in rows:
        print(colorize(f"    {name:<14} configured '{configured}', reports '{live}'", YELLOW))
    print(colorize("  --gpu-type and -m use what nvidia-smi reports; update or remove "
                   "these gpu_type (and gpu_mem_gb/num_gpus) entries.", YELLOW))


def print_detail_table(results: list[ServerResult], free_only: bool = False) -> None:
    """One row per GPU."""
    header = (
        f"{'SERVER':<14} {'GPU':>3}  {'GPU NAME':<24}  "
        f"{'MEM USED':>9}  {'MEM TOT':>7}  {'UTIL':>5}  "
        f"{'STATUS':<8}  {'USER':<12}  {'PROCESS / SCRIPT'}"
    )
    sep = "─" * len(header)
    print(f"\n{BOLD}{header}{RESET}")
    print(sep)

    for r in sorted(results, key=lambda x: (not x.reachable, x.name)):
        if not r.reachable:
            print(f"{r.name:<14} {colorize('UNREACHABLE', RED)}  {r.error[:60]}")
            continue

        for g in sorted(r.gpus, key=lambda x: x.index):
            if free_only and not g.is_free:
                continue

            mem_used_str = f"{mb_to_gb(g.mem_used_mb):>6} GB"
            mem_tot_str  = f"{mb_to_gb(g.mem_total_mb):>4} GB"
            util_str = f"{g.utilization_pct:>3}%" if g.utilization_pct >= 0 else "  N/A"

            if g.is_free:
                status = colorize("FREE", GREEN)
                user_col = ""
                proc_col = ""
                print(
                    f"{r.name:<14} {g.index:>3}  {gpu_model(g.name):<24}  "
                    f"{mem_used_str:>9}  {mem_tot_str:>7}  {util_str:>5}  "
                    f"{status:<16}  {user_col:<12}  {proc_col}"
                )
            else:
                status = colorize("IN USE", RED)
                # Print one line per process (first process on same row as GPU)
                procs = g.processes if g.processes else [GpuProcess(0, "?", 0, "?")]
                first = True
                for p in procs:
                    if first:
                        print(
                            f"{r.name:<14} {g.index:>3}  {gpu_model(g.name):<24}  "
                            f"{mem_used_str:>9}  {mem_tot_str:>7}  {util_str:>5}  "
                            f"{status:<16}  {colorize(p.user, CYAN):<20}  {p.command}"
                        )
                        first = False
                    else:
                        print(
                            f"{'':14} {'':>3}  {'':24}  "
                            f"{'':>9}  {'':>7}  {'':>5}  "
                            f"{'':16}  {colorize(p.user, CYAN):<20}  {p.command}"
                        )

    print(sep)


def print_totals(results: list[ServerResult]) -> None:
    reachable = [r for r in results if r.reachable]
    total_gpus  = sum(r.total_gpus for r in reachable)
    free_gpus   = sum(r.free_gpus  for r in reachable)
    total_vram  = sum(r.total_vram_mb for r in reachable)
    free_vram   = sum(r.free_vram_mb  for r in reachable)
    unreachable = len(results) - len(reachable)

    print(
        f"\n{BOLD}Totals:{RESET}  "
        f"{colorize(str(free_gpus), GREEN)} / {total_gpus} GPUs free  |  "
        f"{mb_to_gb(free_vram)} / {mb_to_gb(total_vram)} GB VRAM free  |  "
        f"{colorize(str(unreachable), RED) if unreachable else colorize('0', GREEN)} servers unreachable\n"
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The flags shared by `gpu-status` and `task gpus`."""
    parser.add_argument("-d", "--detail", action="store_true",
                        help="One row per GPU instead of one per server")
    parser.add_argument("-f", "--free", action="store_true",
                        help="Only servers / GPUs with free capacity")
    parser.add_argument("-H", "--hosts", metavar="SPEC",
                        help="Hosts, groups or tags to query (comma separated)")
    parser.add_argument("--workers", metavar="N", type=int, default=16,
                        help="Parallel SSH connections (default: 16)")
    parser.add_argument("--debug", action="store_true",
                        help="Print raw SSH output for each server")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="gpu-status",
        description="Check GPU availability across configured GPU servers."
    )
    add_arguments(parser)
    run(parser.parse_args(argv))


def run(args: argparse.Namespace) -> None:
    try:
        cfg = load_config()
        targets = cfg.select(args.hosts)
    except ConfigError as e:
        sys.exit(f"gpu-status: {e}")
    transport = Transport(cfg)

    print(f"{BOLD}Querying {len(targets)} server(s)...{RESET}  "
          f"{DIM}network: {transport.describe_mode()}{RESET}", flush=True)

    done = [0]

    def query_one(host):
        r = query_host(transport, host, debug=args.debug)
        done[0] += 1
        status = "ok" if r.reachable else f"ERR: {r.error[:60]}"
        print(f"  [{done[0]:>2}/{len(targets)}] {host.name:<16} {status}", flush=True)
        return r

    results = list(transport.fan_out(targets, query_one, workers=args.workers).values())

    if args.detail:
        print_detail_table(results, free_only=args.free)
        print_label_warnings(results, cfg)
    else:
        print_summary_table(results, cfg, free_only=args.free)

    print_totals(results)


if __name__ == "__main__":
    main()
