"""
Host configuration for task-spooler.

Hosts are read from ~/.task-spooler/hosts.toml (override with the
TASK_SPOOLER_CONFIG environment variable). When the file is absent a built-in
default reproducing the original hardcoded ECS server list is used.
"""

import ipaddress
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib

STATE_DIR = Path.home() / ".task-spooler"
DEFAULT_CONFIG_PATH = STATE_DIR / "hosts.toml"

# ── Built-in server catalogue (the original hardcoded list) ───────────────────

DEFAULT_SERVERS = [
    "cuda-small0", "cuda-small1",
    "cuda00", "cuda01", "cuda02", "cuda03", "cuda04", "cuda05", "cuda06",
    "cuda07", "cuda08", "cuda09", "cuda10", "cuda11", "cuda12", "cuda13",
    "cuda14", "cuda15", "cuda16", "cuda17", "cuda18", "cuda19", "cuda20",
    "cuda21", "cuda22",
    "gryphon", "red-tomatoes", "piccolo", "the-villa", "bordeaux",
]

#               gpu_type        num_gpus gpu_mem_gb cpu_cores sys_mem_gb
DEFAULT_SERVER_INFO = {
    "cuda-small0":  ("RTX A4000",    4, 16, 16,  192),
    "cuda-small1":  ("RTX 4000 Ada", 3, 20, 16,  144),
    "cuda00":       ("RTX 6000",     2, 24, 32,  96),
    "cuda01":       ("RTX A5000",    2, 24, 16,  64),
    "cuda02":       ("RTX A5000",    2, 24, 16,  64),
    "cuda03":       ("RTX A5000",    2, 24, 16,  64),
    "cuda04":       ("RTX A5000",    2, 24, 16,  64),
    "cuda05":       ("RTX A5000",    3, 24, 24,  144),
    "cuda06":       ("RTX A5000",    3, 24, 16,  144),
    "cuda07":       ("RTX A5000",    3, 24, 24,  144),
    "cuda08":       ("RTX A5000",    3, 24, 24,  144),
    "cuda09":       ("RTX 6000",     3, 24, 32,  96),
    "cuda10":       ("RTX A5000",    3, 24, 24,  128),
    "cuda11":       ("RTX A5000",    3, 24, 24,  112),
    "cuda12":       ("RTX 6000",     3, 24, 32,  96),
    "cuda13":       ("RTX 6000",     3, 24, 32,  96),
    "cuda14":       ("A40",          4, 48, 64,  256),
    "cuda15":       ("A40",          4, 48, 64,  256),
    "cuda16":       ("RTX A6000",    3, 48, 32,  192),
    "cuda17":       ("RTX 6000",     3, 24, 24,  144),
    "cuda18":       ("RTX A6000",    3, 48, 32,  192),
    "cuda19":       ("L4",           4, 24, 64,  256),
    "cuda20":       ("L40S",         4, 48, 64,  384),
    "cuda21":       ("RTX 6000 Ada", 2, 48, 48,  256),
    "cuda22":       ("RTX 6000 Ada", 2, 48, 48,  256),
    "gryphon":      ("RTX A6000",    1, 48, 64,  192),
    "red-tomatoes": ("RTX A6000",    2, 48, 32,  96),
    "piccolo":      ("RTX A5000",    2, 24, 20,  128),
    "the-villa":    ("RTX A6000",    1, 48, 64,  192),
    "bordeaux":     ("RTX 6000",     1, 24, 20,  64),
}

INFO_FIELDS = ("gpu_type", "num_gpus", "gpu_mem_gb", "cpu_cores", "sys_mem_gb")

# ── Model ─────────────────────────────────────────────────────────────────────


class ConfigError(Exception):
    pass


@dataclass
class Host:
    name: str
    kind: str = "ssh"                 # ssh | local
    hostname: Optional[str] = None    # None → ssh resolves the alias itself
    user: Optional[str] = None
    port: Optional[int] = None
    jump: Optional[str] = None        # name of a [jumps.X] chain
    role: str = "worker"              # worker | jump
    tags: list[str] = field(default_factory=list)
    group: Optional[str] = None       # first [groups.X] the host is listed in
    path_map: list[tuple[str, str]] = field(default_factory=list)
    cpu_slots: int = 0
    enabled: bool = True
    gpu_type: Optional[str] = None
    num_gpus: Optional[int] = None
    gpu_mem_gb: Optional[float] = None
    cpu_cores: Optional[int] = None
    sys_mem_gb: Optional[float] = None

    @property
    def alias(self) -> str:
        """Name used as the `Host` alias in the generated ssh config."""
        return self.name

    @property
    def is_local(self) -> bool:
        return self.kind == "local"


@dataclass
class Config:
    hosts: dict[str, Host]
    jumps: dict[str, list[str]]       # chain name → ordered hop host names
    mode: str = "direct"              # auto | direct | jump
    probe: Optional[str] = None
    identity_file: Optional[str] = None
    control_persist: str = "10m"
    jump_control_persist: str = "12h"
    extra_options: list[str] = field(default_factory=list)
    source: Optional[Path] = None     # None → built-in default
    mtime: float = 0.0

    @property
    def source_label(self) -> str:
        return str(self.source) if self.source else "built-in default"

    def workers(self) -> list[Host]:
        return [h for h in self.hosts.values() if h.enabled and h.role == "worker"]

    def get(self, name: str) -> Host:
        try:
            return self.hosts[name]
        except KeyError:
            raise ConfigError(f"unknown host '{name}'") from None

    def select(self, spec: Optional[str]) -> list[Host]:
        """
        Resolve a comma-separated list of host names, group names or tags to
        enabled worker hosts (in config order). Empty/None selects all workers.
        """
        workers = self.workers()
        if not spec:
            return workers
        wanted: set[str] = set()
        for item in (s.strip() for s in spec.split(",")):
            if not item:
                continue
            matched = [h.name for h in workers if h.name == item or item in h.tags]
            if not matched:
                raise ConfigError(f"'{item}' matches no enabled host, group or tag")
            wanted.update(matched)
        return [h for h in workers if h.name in wanted]

    def chain(self, host: Host) -> list[Host]:
        """Ordered jump hops for a host (empty if it has no jump)."""
        if not host.jump:
            return []
        return [self.hosts[n] for n in self.jumps[host.jump]]

    def _path_match(self, host: Host, local_path: str) -> Optional[tuple[str, str]]:
        best = None
        for local, remote in host.path_map:
            l = local.rstrip("/") or "/"
            if l == "/" or local_path == l or local_path.startswith(l + "/"):
                if best is None or len(l) > len(best[0]):
                    best = (l, remote)
        return best

    def map_path(self, host: Host, local_path: str) -> str:
        """Longest-prefix match of local_path against the host's path_map."""
        best = self._path_match(host, local_path)
        if best is None:
            return local_path
        l, remote = best
        rest = local_path[len(l):].strip("/") if l != "/" else local_path.strip("/")
        remote = remote.rstrip("/")
        return f"{remote}/{rest}" if rest else (remote or "/")

    def is_mapped(self, host: Host, local_path: str) -> bool:
        return self._path_match(host, local_path) is not None


# ── Loading ───────────────────────────────────────────────────────────────────


def config_path() -> Path:
    env = os.environ.get("TASK_SPOOLER_CONFIG")
    return Path(env).expanduser() if env else DEFAULT_CONFIG_PATH


def builtin_config() -> Config:
    hosts: dict[str, Host] = {}
    for name in DEFAULT_SERVERS:
        gpu_type, num_gpus, mem, cores, sysmem = DEFAULT_SERVER_INFO[name]
        hosts[name] = Host(
            name=name, tags=["ecs"], group="ecs", gpu_type=gpu_type, num_gpus=num_gpus,
            gpu_mem_gb=mem, cpu_cores=cores, sys_mem_gb=sysmem,
        )
    return Config(hosts=hosts, jumps={}, mode="direct")


def load_config(path: Optional[Path] = None) -> Config:
    path = path or config_path()
    if not path.exists():
        return builtin_config()
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None
    cfg = parse_config(data)
    cfg.source = path
    cfg.mtime = path.stat().st_mtime
    return cfg


def _parse_spec(spec: str) -> tuple[Optional[str], str, Optional[int]]:
    """'user@host[:port]' → (user, host, port)."""
    user = None
    if "@" in spec:
        user, spec = spec.rsplit("@", 1)
    port = None
    if spec.startswith("["):                       # [ipv6]:port
        host, _, rest = spec[1:].partition("]")
        if rest.startswith(":"):
            port = int(rest[1:])
    elif spec.count(":") == 1:
        host, p = spec.split(":")
        port = int(p)
    else:
        host = spec
    return user, host, port


def _spec_alias(hostname: str, taken: set[str]) -> str:
    try:
        ipaddress.ip_address(hostname)
        base = "ts-" + hostname.replace(".", "-").replace(":", "-")
    except ValueError:
        base = "ts-" + hostname.split(".")[0]
    alias, n = base, 2
    while alias in taken:
        alias, n = f"{base}-{n}", n + 1
    return alias


_HOST_KEYS = {
    "kind", "hostname", "user", "port", "jump", "role", "tags", "path_map",
    "cpu_slots", "enabled", "domain", *INFO_FIELDS,
}


def _path_map(value, where: str) -> list[tuple[str, str]]:
    out = []
    for entry in value or []:
        try:
            out.append((os.path.expanduser(entry["local"]), entry["remote"]))
        except (TypeError, KeyError):
            raise ConfigError(f"{where}: path_map entries need 'local' and 'remote'") from None
    return out


def parse_config(data: dict) -> Config:
    net = data.get("network", {})
    ssh = data.get("ssh", {})
    groups = data.get("groups", {})
    host_tables = data.get("hosts", {})
    jump_tables = data.get("jumps", {})

    mode = net.get("mode", "auto")
    if mode not in ("auto", "direct", "jump"):
        raise ConfigError(f"network.mode must be auto, direct or jump (got '{mode}')")

    # 1. Collect settings per host name: group defaults first, then [hosts.X].
    settings: dict[str, dict] = {}
    order: list[str] = []
    for gname, g in groups.items():
        defaults = {k: v for k, v in g.items() if k != "hosts"}
        unknown = set(defaults) - _HOST_KEYS
        if unknown:
            raise ConfigError(f"groups.{gname}: unknown key(s) {sorted(unknown)}")
        for hname in g.get("hosts", []):
            s = settings.setdefault(hname, {"tags": []})
            if hname not in order:
                order.append(hname)
            s.setdefault("_group", gname)          # first group listed wins
            for k, v in defaults.items():
                if k != "tags":
                    s.setdefault(k, v)
            for t in [gname, *defaults.get("tags", [])]:
                if t not in s["tags"]:
                    s["tags"].append(t)

    for hname, h in host_tables.items():
        unknown = set(h) - _HOST_KEYS
        if unknown:
            raise ConfigError(f"hosts.{hname}: unknown key(s) {sorted(unknown)}")
        s = settings.setdefault(hname, {"tags": []})
        if hname not in order:
            order.append(hname)
        for k, v in h.items():
            if k == "tags":
                s["tags"] += [t for t in v if t not in s["tags"]]
            else:
                s[k] = v

    # 2. Jump chains: names refer to hosts; raw specs become synthetic hosts.
    jumps: dict[str, list[str]] = {}
    synthetic: dict[str, Host] = {}
    spec_aliases: dict[str, str] = {}
    for jname, j in jump_tables.items():
        chain = j.get("chain", [])
        if not chain:
            raise ConfigError(f"jumps.{jname}: chain is empty")
        hops = []
        for elem in chain:
            if elem in settings:
                if "_group" not in settings[elem]:
                    settings[elem].setdefault("role", "jump")
                hops.append(elem)
            elif elem in spec_aliases:
                hops.append(spec_aliases[elem])
            else:
                user, hostname, port = _parse_spec(elem)
                alias = _spec_alias(hostname, set(settings) | set(synthetic))
                synthetic[alias] = Host(
                    name=alias, hostname=hostname, user=user, port=port, role="jump",
                )
                spec_aliases[elem] = alias
                hops.append(alias)
        jumps[jname] = hops

    # 3. Build Host objects.
    hosts: dict[str, Host] = {}
    for hname in order:
        s = settings[hname]
        domain = s.get("domain")
        hostname = s.get("hostname") or (f"{hname}.{domain}" if domain else None)
        kind = s.get("kind", "ssh")
        if kind not in ("ssh", "local"):
            raise ConfigError(f"hosts.{hname}: kind must be ssh or local")
        jump = s.get("jump")
        if jump and jump not in jumps:
            raise ConfigError(f"hosts.{hname}: unknown jump chain '{jump}'")
        role = s.get("role", "worker")
        if role not in ("worker", "jump"):
            raise ConfigError(f"hosts.{hname}: role must be worker or jump")
        hosts[hname] = Host(
            name=hname, kind=kind, hostname=hostname, user=s.get("user"),
            port=s.get("port"), jump=None if kind == "local" else jump, role=role,
            tags=list(s.get("tags", [])), group=s.get("_group"),
            path_map=_path_map(s.get("path_map"), f"hosts.{hname}"),
            cpu_slots=int(s.get("cpu_slots", 0)), enabled=bool(s.get("enabled", True)),
            **{k: s.get(k) for k in INFO_FIELDS},
        )
    hosts.update(synthetic)

    # A hop must not itself route through the chain it belongs to.
    for jname, hops in jumps.items():
        for hop in hops:
            if hosts[hop].jump:
                raise ConfigError(f"jumps.{jname}: hop '{hop}' must not set its own jump")

    return Config(
        hosts=hosts, jumps=jumps, mode=mode, probe=net.get("probe"),
        identity_file=ssh.get("identity_file"),
        control_persist=str(ssh.get("control_persist", "10m")),
        jump_control_persist=str(ssh.get("jump_control_persist", "12h")),
        extra_options=list(ssh.get("extra_options", [])),
    )


# ── Template for --init-config ────────────────────────────────────────────────


def config_template(user: str = "woodj4") -> str:
    info_blocks = []
    for name in DEFAULT_SERVERS:
        gpu_type, num_gpus, mem, cores, sysmem = DEFAULT_SERVER_INFO[name]
        info_blocks.append(
            f"[hosts.{name}]\n"
            f'gpu_type = "{gpu_type}"\n'
            f"num_gpus = {num_gpus}\n"
            f"gpu_mem_gb = {mem}\n"
            f"cpu_cores = {cores}\n"
            f"sys_mem_gb = {sysmem}\n"
        )
    host_list = ",\n    ".join(f'"{n}"' for n in DEFAULT_SERVERS)
    return f'''\
# task-spooler host configuration.
# Docs: see the "Configuration" section of the README.

[network]
# auto   – probe the network: connect directly when on campus, via the jump
#          chain otherwise (result cached for 5 minutes)
# direct – always connect directly
# jump   – always go through the jump chain
mode = "auto"
# probe = "cuda00.ecs.vuw.ac.nz"   # default: first worker that has a jump

[ssh]
# identity_file = "~/.ssh/id_ed25519"
control_persist = "10m"        # how long worker connections stay open
jump_control_persist = "12h"   # how long jump-hop connections stay open
extra_options = []             # raw "Key=Value" ssh options

# Off-campus route: laptop → entry → barretts → cudaXX.
[jumps.ecs]
chain = ["{user}@entry.ecs.vuw.ac.nz", "{user}@barretts.ecs.vuw.ac.nz"]

[groups.ecs]
user = "{user}"
domain = "ecs.vuw.ac.nz"       # hostname = "<name>.<domain>"
jump = "ecs"
cpu_slots = 0
# Translate local working directories to their path on ECS, e.g. when
# submitting from a laptop checkout that mirrors an ECS directory:
# path_map = [ {{ local = "~/code", remote = "/vol/ecrg-solar/{user}" }} ]
hosts = [
    {host_list},
]

# Run jobs on this machine as well. -G 0 jobs use cpu_slots.
# [hosts.local]
# kind = "local"
# cpu_slots = 2

# Per-host details (used for -m/--min-mem and display). Set enabled = false
# to stop dispatching to a host.
{chr(10).join(info_blocks)}'''
