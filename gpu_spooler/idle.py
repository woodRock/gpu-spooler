"""
GPUs held but idle, read from gpu-detective's history.

Opt-in with a [detective] table in hosts.toml. gpu-spooler never imports
gpu-detective: the contract is the history's SQLite schema (snapshots, gpus,
procs), opened read-only. A missing or unreadable history just means no
idle information.
"""

import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

IDLE_UTIL_PCT = 5       # same threshold as gpu-detective
STALE = 600             # a host's latest snapshot must be this recent to count
MIN_LOOKBACK = 6 * 3600


@dataclass
class IdleGpu:
    host: str
    index: int
    idle_seconds: float
    users: list[str]
    capped: bool = False    # idle for at least the lookback: idle_seconds is a lower bound


def idle_gpus(path: Path, min_seconds: float, now: Optional[float] = None) -> dict[tuple[str, int], IdleGpu]:
    """
    (host, gpu index) → IdleGpu for every GPU that, in the latest sweep, holds
    processes at under IDLE_UTIL_PCT utilisation and has done so, with the same
    users, in every sweep for at least min_seconds.
    """
    now = time.time() if now is None else now
    lookback = max(MIN_LOOKBACK, 2 * min_seconds)
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
    except sqlite3.Error:
        return {}
    try:
        snaps = con.execute(
            "SELECT id, ts, host, reachable FROM snapshots WHERE ts >= ? ORDER BY host, ts",
            (now - lookback,)).fetchall()
        if not snaps:
            return {}
        lo = min(s[0] for s in snaps)
        util: dict[int, dict[int, int]] = defaultdict(dict)
        for sid, idx, u in con.execute(
                "SELECT snapshot_id, idx, util_pct FROM gpus WHERE snapshot_id >= ?", (lo,)):
            util[sid][idx] = u
        users: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
        for sid, idx, user in con.execute(
                "SELECT snapshot_id, gpu_idx, user FROM procs WHERE snapshot_id >= ?", (lo,)):
            users[sid][idx].add(user)
    except sqlite3.Error:
        return {}
    finally:
        con.close()

    series: dict[str, list[tuple]] = defaultdict(list)
    for sid, ts, host, reachable in snaps:
        series[host].append((sid, ts, bool(reachable)))

    def idle(sid: int, idx: int) -> bool:
        u = util[sid].get(idx)
        return u is not None and 0 <= u < IDLE_UTIL_PCT and bool(users[sid][idx])

    out = {}
    for host, ss in series.items():
        sid, ts, reachable = ss[-1]
        if not reachable or ts < now - STALE:
            continue
        for idx in util[sid]:
            if not idle(sid, idx):
                continue
            holders = users[sid][idx]
            since, capped = ts, True
            for psid, pts, preach in reversed(ss[:-1]):
                if not preach or not idle(psid, idx) or users[psid][idx] != holders:
                    capped = False
                    break
                since = pts
            if ts - since >= min_seconds:
                out[(host, idx)] = IdleGpu(host, idx, ts - since, sorted(holders),
                                           capped and len(ss) > 1)
    return out


def fmt_idle(g: IdleGpu) -> str:
    s = int(g.idle_seconds)
    text = f"{s // 60}m" if s < 3600 else f"{s // 3600}h{s % 3600 // 60:02d}m"
    return f"idle {text}{'+' if g.capped else ''}"
