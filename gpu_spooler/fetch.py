"""
Copying outputs back from the servers: `task fetch PATH...`.

Paths resolve the way jobs' working directories do (Config.remote_dir): from a
laptop checkout of a git repo, a relative PATH is the same path in the repo's
copy under remote_root, or wherever a path_map entry points. REMOTE:LOCAL names
both sides explicitly, for anything outside the repo copy. Directories are
merged into the local directory of the same name.

Safety rules:
  - a relative PATH must stay inside the repo (REMOTE:LOCAL for anything else);
  - symlinks are copied as links, never followed, and a link pointing outside
    the destination is skipped (tarfile's 'data' filter);
  - a local file is replaced only if it is unchanged since the last fetch wrote
    it, or tracked by git and unmodified (git still has it). Anything else is
    kept and reported, unless force=True;
  - fetch never deletes local files that are missing on the server.
"""

import base64
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from typing import Optional

from gpu_spooler.config import STATE_DIR, Config, Host
from gpu_spooler.repo import find_repo
from gpu_spooler.transport import Transport

MANIFEST = STATE_DIR / "fetched.json"   # local path → [size, mtime_ns] as fetch left it
LIST_TIMEOUT = 300
STEP = "TS_FETCH"

# Runs on the server: describe each requested path (files with their SHA-256, links).
_LIST_PY = r'''
import base64, hashlib, json, os, sys
paths = json.loads(base64.b64decode(sys.argv[1]))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
out = []
for path in paths:
    p = os.path.expanduser(path)
    if os.path.islink(p):
        out.append({"kind": "link", "target": os.readlink(p)})
    elif os.path.isfile(p):
        out.append({"kind": "file", "sha": sha(p)})
    elif os.path.isdir(p):
        files, links = {}, {}
        for root, dirs, names in os.walk(p):        # does not descend into linked dirs
            for n in dirs + names:
                full = os.path.join(root, n)
                rel = os.path.relpath(full, p)
                if os.path.islink(full):
                    links[rel] = os.readlink(full)
                elif os.path.isfile(full):
                    files[rel] = sha(full)
        out.append({"kind": "dir", "files": files, "links": links})
    else:
        out.append({"kind": "missing"})
print("TS_FETCH " + json.dumps(out))
'''

# Runs on the server: a gzipped tar of the given members of base, links as links.
_TAR_PY = r'''
import base64, json, os, sys, tarfile
req = json.loads(base64.b64decode(sys.argv[1]))
os.chdir(os.path.expanduser(req["base"]))
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as tf:
    for rel in req["members"]:
        tf.add(rel, recursive=False)
'''


class FetchError(Exception):
    pass


def _py_script(program: str, payload) -> str:
    arg = base64.b64encode(json.dumps(payload).encode()).decode()
    return (f'command -v python3 >/dev/null 2>&1 || {{ echo "python3 not found" >&2; exit 3; }}\n'
            f"python3 - {arg} <<'TS_PY'\n{program}\nTS_PY\n")


# ── Resolving paths ──────────────────────────────────────────────────────────


@dataclass
class Item:
    spec: str          # as given
    local: str         # absolute local path
    remote: str        # path on the server


@dataclass
class Plan:
    item: Item
    kind: str                                       # file | dir | link | missing
    new: list[str] = field(default_factory=list)      # relative to the item (or "" for a file)
    update: list[str] = field(default_factory=list)
    same: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)     # local changes, not replaced
    conflict: list[str] = field(default_factory=list) # a directory where a file would go
    links: dict[str, str] = field(default_factory=dict)

    @property
    def to_fetch(self) -> list[str]:
        return self.new + self.update


def _inside(path: str, root: str) -> bool:
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root.rstrip("/") + "/")


def _is_explicit(spec: str) -> bool:
    head, sep, _ = spec.partition(":")
    return bool(sep) and head.startswith(("/", "~"))


def resolve(cfg: Config, hosts: list[Host], specs: list[str], cwd: str) -> tuple[list[Host], list[Item]]:
    """
    The hosts to fetch from (one storage group, tried in order) and each spec's
    local and remote path. Raises FetchError when a path cannot be mapped.
    """
    hosts = [h for h in hosts if not h.is_local]
    if not hosts:
        raise FetchError("no remote host selected")
    groups: dict[tuple, list[Host]] = {}
    for h in hosts:
        groups.setdefault((h.group or h.name, h.remote_root), []).append(h)
    if len(groups) > 1:
        names = ", ".join(sorted(g for g, _ in groups))
        raise FetchError(f"the selected hosts span several storage groups ({names}); "
                         "pick one with -H")
    group = next(iter(groups.values()))
    host = group[0]
    local_repo = find_repo(cwd)

    items = []
    for spec in specs:
        if _is_explicit(spec):
            remote, _, local = spec.partition(":")
            local = os.path.abspath(os.path.join(cwd, os.path.expanduser(local or ".")))
            items.append(Item(spec, local, remote.rstrip("/") or "/"))
            continue
        local = os.path.abspath(os.path.join(cwd, os.path.expanduser(spec)))
        if cfg.is_mapped(host, local):
            items.append(Item(spec, local, cfg.map_path(host, local)))
            continue
        if local_repo is None or not host.remote_root:
            raise FetchError(f"{spec}: not inside a git repo with a remote copy and no path_map "
                             "entry covers it; use REMOTE:LOCAL")
        if not _inside(local, local_repo.root):
            raise FetchError(f"{spec} is outside the repo {local_repo.root}; use REMOTE:LOCAL "
                             "to fetch it")
        items.append(Item(spec, local, cfg.remote_dir(host, local, local_repo.root)))
    return group, items


# ── Comparing ────────────────────────────────────────────────────────────────


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _stamp(path: str) -> list[int]:
    st = os.lstat(path)
    return [st.st_size, st.st_mtime_ns]


def load_manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text())
    except (OSError, ValueError):
        return {}


def save_manifest(m: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_name(MANIFEST.name + ".tmp")
    tmp.write_text(json.dumps(m))
    tmp.replace(MANIFEST)


def git_clean_tracked(paths: list[str]) -> set[str]:
    """Those of paths that git tracks and that are unmodified (git can restore them)."""
    out: set[str] = set()
    by_root: dict[str, list[str]] = {}
    for p in paths:
        r = find_repo(os.path.dirname(p) if not os.path.isdir(p) else p)
        if r:
            by_root.setdefault(r.root, []).append(p)
    for root, ps in by_root.items():
        rels = [os.path.relpath(os.path.realpath(p), os.path.realpath(root)) for p in ps]

        def git(*args: str) -> set[str]:
            r = subprocess.run(["git", "-C", root, *args, "--", *rels],
                               capture_output=True, text=True)
            return {x for x in r.stdout.split("\0") if x} if r.returncode == 0 else set()

        tracked = git("ls-files", "-z")
        modified = git("diff", "--name-only", "-z", "HEAD")
        real_root = os.path.realpath(root)
        for p, rel in zip(ps, rels):
            if rel in tracked and rel not in modified:
                out.add(p)
                out.add(os.path.join(real_root, rel))
    return out


def plan_item(item: Item, desc: dict, manifest: dict, force: bool) -> Plan:
    plan = Plan(item, desc["kind"])
    if desc["kind"] == "missing":
        return plan
    if desc["kind"] != "dir" and os.path.isdir(item.local) and not os.path.islink(item.local):
        item.local = os.path.join(item.local, os.path.basename(item.remote))  # like scp
    if desc["kind"] == "dir":
        entries = {**{r: ("file", s) for r, s in desc["files"].items()},
                   **{r: ("link", t) for r, t in desc["links"].items()}}
        base = item.local
    else:
        entries = {"": (desc["kind"], desc.get("sha") or desc.get("target"))}
        base = None
    dests = {rel: (os.path.join(base, rel) if base else item.local) for rel in entries}
    clean = git_clean_tracked([d for d in dests.values() if os.path.lexists(d)])
    for rel, (kind, ref) in sorted(entries.items()):
        dest = dests[rel]
        if kind == "link":
            plan.links[rel] = ref
        if not os.path.lexists(dest):
            plan.new.append(rel)
            continue
        if os.path.isdir(dest) and not os.path.islink(dest):
            plan.conflict.append(rel)
            continue
        if kind == "file" and not os.path.islink(dest) and _sha(dest) == ref:
            plan.same.append(rel)
            continue
        if kind == "link" and os.path.islink(dest) and os.readlink(dest) == ref:
            plan.same.append(rel)
            continue
        replaceable = (force or manifest.get(dest) == _stamp(dest)
                       or dest in clean or os.path.realpath(dest) in clean)
        (plan.update if replaceable else plan.kept).append(rel)
    return plan


# ── Fetching ─────────────────────────────────────────────────────────────────


def _describe(transport: Transport, hosts: list[Host], items: list[Item]) -> tuple[Host, list[dict]]:
    """Ask the first reachable host of the group to describe every item."""
    last = ""
    for host in hosts:
        rc, out, err = transport.run(host, _py_script(_LIST_PY, [i.remote for i in items]),
                                     timeout=LIST_TIMEOUT)
        line = next((l for l in out.splitlines() if l.startswith(STEP + " ")), None)
        if rc == 0 and line:
            return host, json.loads(line[len(STEP) + 1:])
        last = (err.strip().splitlines() or [f"exit {rc}"])[-1]
        if rc != 255:          # reachable but failed: another host of the group won't help
            break
    raise FetchError(f"could not list the remote paths: {last}")


def _transfer(transport: Transport, host: Host, plan: Plan) -> list[str]:
    """Copy plan.to_fetch into place; returns the members skipped by the tar filter."""
    item = plan.item
    if plan.kind == "dir":
        base_remote, members, dest_root = item.remote, plan.to_fetch, item.local
    else:
        base_remote = os.path.dirname(item.remote) or "/"
        members = [os.path.basename(item.remote)]
        dest_root = os.path.dirname(item.local)
    os.makedirs(dest_root, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".task-fetch-", dir=dest_root)
    skipped: list[str] = []
    try:
        with tempfile.TemporaryFile() as err:
            proc = transport.popen(host, stdout=subprocess.PIPE, stderr=err)
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(_py_script(_TAR_PY, {"base": base_remote,
                                                  "members": members}).encode())
            proc.stdin.close()
            try:
                with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tf:
                    for m in tf:
                        try:
                            tf.extract(m, tmp, filter="data")
                        except tarfile.FilterError as e:
                            skipped.append(f"{m.name} ({e})")
            except tarfile.TarError as e:
                proc.kill()
                err.seek(0)
                detail = err.read().decode(errors="replace").strip()
                raise FetchError(f"{item.spec}: transfer failed: {detail or e}") from None
            if proc.wait() != 0:
                err.seek(0)
                raise FetchError(f"{item.spec}: transfer failed: "
                                 f"{err.read().decode(errors='replace').strip()}")
        for rel in members:
            src = os.path.join(tmp, rel)
            if not os.path.lexists(src):
                continue        # skipped by the filter
            dest = os.path.join(dest_root, rel) if plan.kind == "dir" else item.local
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.islink(dest) or os.path.isfile(dest):
                os.unlink(dest)
            os.replace(src, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return skipped


# ── Which task wrote what (task fetch --after A B ...) ─────────────────────────

WRITE_SLACK = 120   # seconds around a task's run: clock skew and the daemon's 5 s polls


def _named_paths(row) -> list[str]:
    """
    Local paths the task's command names, as an argument or a --flag=value, resolved from its
    working directory. The working directory and anything above it are left out: they name
    every file.
    """
    try:
        words = shlex.split(row["command"])
    except ValueError:
        words = row["command"].split()
    wdir = row["working_dir"]
    out = []
    for w in words:
        value = w.split("=", 1)[1] if "=" in w else w
        if not value or value.startswith("-"):
            continue
        p = os.path.normpath(os.path.join(wdir, os.path.expanduser(value)))
        if not _inside(wdir, p):
            out.append(p)
    return out


def _depth_named(row, path: str) -> int:
    """How deep the command names path (its own name or a directory above it); -1 if it doesn't."""
    return max((len(p.split(os.sep)) for p in _named_paths(row) if _inside(path, p)), default=-1)


def _ran_when_written(row, path: str) -> bool:
    if not row["started_at"] or not row["finished_at"]:
        return False
    mtime = os.lstat(path).st_mtime   # the server's: the tar extraction keeps it
    return row["started_at"] - WRITE_SLACK <= mtime <= row["finished_at"] + WRITE_SLACK


def attribute(rows: list, paths: list[str]) -> tuple[dict[int, list[str]], list[str]]:
    """
    Credit each fetched file to the tasks that wrote it: the tasks whose command names it most
    specifically (`--out results/a` names results/a/x.csv), then, if that leaves more than one,
    those that were running when it was written. Returns ({task id: paths}, the paths still
    credited to more than one task). With one task, it gets everything.
    """
    by_task: dict[int, list[str]] = {r["id"]: [] for r in rows}
    shared = []
    for path in paths:
        depth = {r["id"]: _depth_named(r, path) for r in rows}
        deepest = max(depth.values(), default=-1)
        cands = [r for r in rows if depth[r["id"]] == deepest]
        if len(cands) > 1:
            cands = [r for r in cands if _ran_when_written(r, path)] or cands
        if len(cands) > 1:
            shared.append(path)
        for r in cands:
            by_task[r["id"]].append(path)
    return by_task, shared


@dataclass
class FetchResult:
    plans: list[Plan]
    host: Host
    skipped: dict[str, list[str]] = field(default_factory=dict)   # spec → filtered members


def fetch(cfg: Config, transport: Transport, hosts: list[Host], specs: list[str], cwd: str,
          force: bool = False, dry_run: bool = False) -> FetchResult:
    if not hasattr(tarfile, "data_filter"):
        raise FetchError("--fetch needs Python ≥ 3.12 (or 3.10.12 / 3.11.4) for safe extraction")
    group, items = resolve(cfg, hosts, specs, cwd)
    host, descs = _describe(transport, group, items)
    manifest = load_manifest()
    plans = [plan_item(i, d, manifest, force) for i, d in zip(items, descs)]
    res = FetchResult(plans, host)
    if dry_run:
        return res
    for plan in plans:
        if not plan.to_fetch:
            continue
        res.skipped[plan.item.spec] = _transfer(transport, host, plan)
        base = plan.item.local if plan.kind == "dir" else None
        for rel in plan.to_fetch:
            dest = os.path.join(base, rel) if base else plan.item.local
            if os.path.lexists(dest):
                manifest[dest] = _stamp(dest)
    save_manifest(manifest)
    return res

