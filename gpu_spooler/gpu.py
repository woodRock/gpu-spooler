"""
GPU querying shared by the daemon and `gpu-status`.

One script runs nvidia-smi on a host and prints sections delimited by sentinel
lines; parse_output() turns that into GpuInfo objects.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from gpu_spooler.config import Host
from gpu_spooler.transport import Transport, auth_hint

BUSY_MEM_MB = 256       # a GPU using at least this much VRAM counts as busy
QUERY_TIMEOUT = 25

REMOTE_CMD = r"""
if ! command -v nvidia-smi >/dev/null 2>&1; then echo "NO_NVIDIA_SMI"; exit 0; fi
GPU_OUT=$(nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu,compute_mode \
    --format=csv,noheader,nounits 2>/dev/null) || { echo "ERROR: nvidia-smi failed"; exit 1; }
PMON_OUT=$(nvidia-smi pmon -c 1 2>/dev/null)
APPS_OUT=$(nvidia-smi --query-compute-apps=gpu_index,pid --format=csv,noheader 2>/dev/null)
printf 'GPU_INFO\n%s\nPMON_INFO\n%s\nAPPS_INFO\n%s\nPROC_INFO\n' "$GPU_OUT" "$PMON_OUT" "$APPS_OUT"
PIDS=$( { printf '%s\n' "$PMON_OUT" | awk '!/^#/ && $2 ~ /^[0-9]+$/ {print $2}'
          printf '%s\n' "$APPS_OUT" | awk -F', *' '$2 ~ /^[0-9]+$/ {print $2}'; } | sort -u | tr '\n' ',')
PIDS="${PIDS%,}"
if [ -n "$PIDS" ]; then
    ps -o pid=,user= -p "$PIDS" 2>/dev/null || true
fi
printf 'END\n'
"""


def gpu_model(name: str) -> str:
    """A GPU name as compared and shown: nvidia-smi's name without a leading 'NVIDIA '."""
    name = " ".join(name.split())
    return name[7:] if name.lower().startswith("nvidia ") else name


def gpu_type_matches(spec: Optional[str], name: str) -> bool:
    """
    Whether a GPU called `name` (by nvidia-smi) satisfies --gpu-type `spec`: one of the
    comma-separated models, compared exactly, ignoring case and a leading 'NVIDIA '.
    Exact, not substring: 'RTX 6000' must not also pick the 'RTX 6000 Ada Generation'.
    """
    if not spec:
        return True
    model = gpu_model(name).lower()
    return any(gpu_model(t).lower() == model for t in spec.split(",") if t.strip())


@dataclass
class GpuProcess:
    pid: int
    user: str
    mem_used_mb: int
    command: str  # truncated command/script name


@dataclass
class GpuInfo:
    index: int
    name: str
    mem_used_mb: int
    mem_free_mb: int
    mem_total_mb: int
    utilization_pct: int          # -1 if unknown
    processes: list[GpuProcess] = field(default_factory=list)

    @property
    def is_free(self) -> bool:
        # Free only if no process is listed AND VRAM usage is very low
        # (zombie contexts can hold VRAM without showing up in pmon).
        return len(self.processes) == 0 and self.mem_used_mb < BUSY_MEM_MB

    @property
    def mem_total_gb(self) -> float:
        """Decimal GB, as GPUs are marketed (an A40 reads ~48.3)."""
        return self.mem_total_mb * 1.048576 / 1000


@dataclass
class ServerResult:
    name: str
    reachable: bool
    error: str = ""
    gpus: list[GpuInfo] = field(default_factory=list)

    @property
    def total_gpus(self) -> int:
        return len(self.gpus)

    @property
    def free_gpus(self) -> int:
        return sum(1 for g in self.gpus if g.is_free)

    @property
    def busy_gpus(self) -> int:
        return self.total_gpus - self.free_gpus

    @property
    def total_vram_mb(self) -> int:
        return sum(g.mem_total_mb for g in self.gpus)

    @property
    def free_vram_mb(self) -> int:
        return sum(g.mem_free_mb for g in self.gpus)

    @property
    def models(self) -> list[str]:
        """Distinct GPU models on this host, in index order."""
        return list(dict.fromkeys(gpu_model(g.name) for g in self.gpus))


def parse_output(raw: str) -> list[GpuInfo]:
    """Parse the structured output from REMOTE_CMD into GpuInfo objects."""
    SENTINELS = {"GPU_INFO", "PMON_INFO", "APPS_INFO", "PROC_INFO"}
    sections: dict[str, list[str]] = {s: [] for s in SENTINELS}
    current = None
    for line in raw.splitlines():
        stripped = line.rstrip("\r")          # tolerate \r\n from some SSH configs
        if stripped in SENTINELS:
            current = stripped
        elif stripped == "END":
            break
        elif current and stripped.strip():
            sections[current].append(stripped)

    gpus: dict[int, GpuInfo] = {}
    for line in sections["GPU_INFO"]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            idx = int(parts[0])
            util = int(parts[5]) if parts[5].lstrip("-").isdigit() else -1
            gpus[idx] = GpuInfo(
                index=idx, name=parts[1], mem_used_mb=int(parts[2]),
                mem_free_mb=int(parts[3]), mem_total_mb=int(parts[4]),
                utilization_pct=util,
            )
        except (ValueError, IndexError):
            continue

    # pid -> (gpu_idx, fb_mb, cmd_name). pmon's column set varies between
    # driver versions (jpg/ofa were added), so take fb and command from the end.
    pid_to_gpu: dict[int, tuple[int, int, str]] = {}
    for line in sections["PMON_INFO"]:
        if line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        fb_mb = int(parts[-2]) if len(parts) >= 4 and parts[-2].isdigit() else 0
        cmd = parts[-1] if len(parts) >= 3 else "unknown"
        pid_to_gpu[int(parts[1])] = (int(parts[0]), fb_mb, cmd)

    for line in sections["APPS_INFO"]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            pid_to_gpu.setdefault(int(parts[1]), (int(parts[0]), 0, "unknown"))

    pid_user: dict[int, str] = {}
    for line in sections["PROC_INFO"]:
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            pid_user[int(parts[0])] = parts[1]

    for pid, (gpu_idx, fb_mb, cmd) in pid_to_gpu.items():
        if gpu_idx in gpus:
            gpus[gpu_idx].processes.append(GpuProcess(
                pid=pid, user=pid_user.get(pid, "?"), mem_used_mb=fb_mb, command=cmd[:60]))

    return [gpus[i] for i in sorted(gpus)]


def query_host(transport: Transport, host: Host, debug: bool = False) -> ServerResult:
    rc, out, err = transport.run(host, REMOTE_CMD, timeout=QUERY_TIMEOUT)
    if debug:
        print(f"\n--- RAW OUTPUT FROM {host.name} ---")
        print(out or "(empty stdout)")
        if err:
            print(f"STDERR: {err.strip()}")
        print(f"--- END {host.name} (exit={rc}) ---\n")
    if out.strip() == "NO_NVIDIA_SMI":
        # A local machine (or a configured CPU host) without GPUs is still usable.
        if host.is_local or host.cpu_slots > 0:
            return ServerResult(name=host.name, reachable=True)
        return ServerResult(name=host.name, reachable=False, error="nvidia-smi not found")
    if rc != 0 and not out:
        msg = err.strip().splitlines()[-1] if err.strip() else f"ssh exited {rc}"
        hint = auth_hint(err)
        return ServerResult(name=host.name, reachable=False,
                            error=f"{msg} ({hint})" if hint else msg)
    errline = next((l for l in out.splitlines() if l.startswith("ERROR:")), None)
    if errline:
        return ServerResult(name=host.name, reachable=False, error=errline)
    gpus = parse_output(out)
    if not gpus and "GPU_INFO" not in out:
        return ServerResult(name=host.name, reachable=False, error="no nvidia-smi output")
    return ServerResult(name=host.name, reachable=True, gpus=gpus)


def query_hosts(transport: Transport, hosts: Iterable[Host], workers: int = 16,
                debug: bool = False) -> dict[str, ServerResult]:
    return transport.fan_out(hosts, lambda h: query_host(transport, h, debug), workers)


def result_from_dict(d: dict) -> ServerResult:
    """Inverse of asdict(ServerResult), for reading gpu_snapshot.json."""
    gpus = []
    for g in d.get("gpus", []):
        procs = [GpuProcess(**p) for p in g.get("processes", [])]
        gpus.append(GpuInfo(**{k: v for k, v in g.items() if k != "processes"}, processes=procs))
    return ServerResult(name=d["name"], reachable=d["reachable"], error=d.get("error", ""),
                        gpus=gpus)


def write_snapshot(path: Path, results: dict[str, ServerResult], mode: str,
                   keep: Optional[Iterable[str]] = None) -> None:
    """
    Write the latest GPU query for other readers (the tui). Hosts not in
    `results` keep their previous entry if listed in `keep`; each host
    records when it was last queried.
    """
    now = time.time()
    hosts: dict = {}
    if keep is not None:
        try:
            old = json.loads(path.read_text()).get("hosts", {})
            keep = set(keep)
            hosts = {n: h for n, h in old.items() if n in keep}
        except (OSError, ValueError, AttributeError):
            pass
    for name, r in results.items():
        hosts[name] = dict(asdict(r), ts=now)
    data = {"ts": now, "mode": mode, "hosts": hosts}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)
