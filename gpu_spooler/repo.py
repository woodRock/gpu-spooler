"""
Keeping a remote copy of a local git repo, for submitting from a machine that
doesn't share a filesystem with the servers (e.g. a laptop).

The remote copy lives at <remote_root>/<repo name> on each host group (ECS
servers share NFS, so one copy serves the whole group). `task sync` pushes a
snapshot of the local working tree - HEAD plus uncommitted changes to tracked
files, via `git stash create`, which doesn't touch the local repo - over the
spooler's own ssh route, then checks it out (detached) on the remote side.
The servers never need credentials for the git origin.
"""

import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Optional

from gpu_spooler.config import Config, Host
from gpu_spooler.transport import Transport

SYNC_REF = "refs/gpu-spooler/sync"
CHECK_TIMEOUT = 20
PUSH_TIMEOUT = 600


def sh_path(path: str) -> str:
    """Quote a remote path for bash, keeping a leading ~/ expandable."""
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


# git may only be on a login shell's PATH (on ECS it's in /usr/pkg/bin), and
# both our `bash -s` scripts and the remote end of `git push` run non-login.
FIND_GIT = ("command -v git >/dev/null 2>&1 || "
            "PATH=\"$(bash -lc 'printf %s \"$PATH\"' </dev/null 2>/dev/null):$PATH\"\n")
RECEIVE_PACK = "bash -lc 'exec git-receive-pack \"$@\"' git-receive-pack"


def _git(root: str, *args: str, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True,
                          timeout=timeout)

# ── Local repo ────────────────────────────────────────────────────────────────


@dataclass
class LocalRepo:
    root: str                 # top-level directory
    head: Optional[str]       # None for a repo without commits
    branch: Optional[str]
    dirty: bool               # uncommitted changes to tracked files
    _snapshot: Optional[str] = None

    @property
    def name(self) -> str:
        return os.path.basename(self.root.rstrip("/"))

    def snapshot(self) -> str:
        """Commit capturing the working tree (HEAD itself when clean)."""
        if self._snapshot is None:
            commit = ""
            if self.dirty:
                r = _git(self.root, "stash", "create", "gpu-spooler sync")
                commit = r.stdout.strip()
            self._snapshot = commit or self.head or ""
        return self._snapshot

    def tree(self) -> str:
        r = _git(self.root, "rev-parse", f"{self.snapshot()}^{{tree}}")
        return r.stdout.strip()

    def describe(self) -> str:
        what = (self.head or "")[:7] + (f" ({self.branch})" if self.branch else "")
        return what + (" + uncommitted changes" if self.dirty else "")


def find_repo(path: str) -> Optional[LocalRepo]:
    """The git repo containing path, or None (also when git is missing)."""
    try:
        r = _git(path, "rev-parse", "--show-toplevel")
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    root = r.stdout.strip()
    head = _git(root, "rev-parse", "-q", "--verify", "HEAD").stdout.strip() or None
    branch = _git(root, "symbolic-ref", "-q", "--short", "HEAD").stdout.strip() or None
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return LocalRepo(root=root, head=head, branch=branch, dirty=bool(status.stdout.strip()))

# ── Where the copies live ─────────────────────────────────────────────────────


@dataclass
class Target:
    """One remote copy: a representative host for a group sharing remote_root."""
    host: Host
    members: list[str]
    path: str

    @property
    def label(self) -> str:
        return self.host.group or self.host.name


def targets(cfg: Config, hosts: list[Host], repo: LocalRepo, local_dir: str) -> list[Target]:
    """
    Remote copies needed for running local_dir's repo on these hosts. Hosts in
    the same group with the same remote_root are assumed to share storage.
    Hosts where a path_map entry applies are left out (that path is yours).
    """
    out: dict[tuple, Target] = {}
    for h in hosts:
        if h.is_local or not h.remote_root or cfg.is_mapped(h, local_dir):
            continue
        key = (h.group or h.name, h.remote_root)
        if key in out:
            out[key].members.append(h.name)
        else:
            out[key] = Target(h, [h.name], cfg.remote_dir(h, repo.root, repo.root))
    return list(out.values())

# ── Checking a remote copy ────────────────────────────────────────────────────


@dataclass
class RemoteState:
    state: str                     # missing | notgit | ok | nogit | unreachable
    tree: str = ""
    head: str = ""
    dirty: int = 0
    error: str = ""


def check_script(path: str) -> str:
    return FIND_GIT + f'''\
command -v git >/dev/null 2>&1 || {{ echo "STATE nogit"; exit 0; }}
P={sh_path(path)}
if [ ! -e "$P" ]; then echo "STATE missing"
elif [ "$(git -C "$P" rev-parse --show-toplevel 2>/dev/null)" != "$(cd "$P" && pwd -P)" ]; then
  echo "STATE notgit"
else
  echo "STATE ok"
  echo "TREE $(git -C "$P" rev-parse -q --verify 'HEAD^{{tree}}' 2>/dev/null)"
  echo "HEAD $(git -C "$P" rev-parse -q --verify HEAD 2>/dev/null)"
  echo "DIRTY $(git -C "$P" status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')"
fi
'''


def check(transport: Transport, target: Target) -> RemoteState:
    rc, out, err = transport.run(target.host, check_script(target.path), timeout=CHECK_TIMEOUT)
    fields = dict(line.split(" ", 1) if " " in line else (line, "")
                  for line in out.splitlines() if line.strip())
    if rc != 0 or "STATE" not in fields:
        msg = (err.strip().splitlines() or [f"exit {rc}"])[-1]
        return RemoteState("unreachable", error=msg)
    try:
        dirty = int(fields.get("DIRTY", "0") or 0)
    except ValueError:
        dirty = 0
    return RemoteState(fields["STATE"], tree=fields.get("TREE", ""),
                       head=fields.get("HEAD", ""), dirty=dirty)


def staleness(repo: LocalRepo, target: Target, st: RemoteState) -> Optional[str]:
    """None if the remote copy matches the local working tree, else why not."""
    where = f"{target.label}:{target.path}"
    if st.state == "unreachable":
        return f"could not check {where}: {st.error}"
    if st.state == "nogit":
        return f"git is not installed on {target.host.name}"
    if st.state == "missing":
        return f"repo '{repo.name}' doesn't exist on {where}; run `task sync`"
    if st.state == "notgit":
        return f"{where} exists but is not a git repo"
    if st.dirty:
        return (f"{where} has {st.dirty} uncommitted change(s) on the server; "
                "`task sync` will refuse to overwrite conflicting files")
    if st.tree != repo.tree():
        return (f"{where} is at {st.head[:7] or 'no commit'}, but you're at "
                f"{repo.describe()}; run `task sync`")
    return None

# ── Syncing ───────────────────────────────────────────────────────────────────


@dataclass
class SyncResult:
    target: Target
    ok: bool
    message: str


def _prepare_script(path: str) -> str:
    return FIND_GIT + f'''\
command -v git >/dev/null 2>&1 || {{ echo "git is not installed" >&2; exit 3; }}
P={sh_path(path)}
if [ ! -e "$P" ]; then
  mkdir -p "$P" && git init -q "$P" || exit 1
elif [ "$(git -C "$P" rev-parse --show-toplevel 2>/dev/null)" != "$(cd "$P" && pwd -P)" ]; then
  if [ -n "$(ls -A "$P")" ]; then echo "$P exists and is not a git repo" >&2; exit 4; fi
  git init -q "$P" || exit 1
fi
echo READY
'''


def _checkout_script(path: str, commit: str) -> str:
    return FIND_GIT + f'''\
cd {sh_path(path)} || exit 1
git checkout -q --detach {shlex.quote(commit)} || exit 5
echo SYNCED
'''


def sync(transport: Transport, repo: LocalRepo, target: Target) -> SyncResult:
    """Push the local working tree snapshot to one target and check it out."""
    commit = repo.snapshot()
    if not commit:
        return SyncResult(target, False, "the local repo has no commits yet")
    host = target.host

    rc, out, err = transport.run(host, _prepare_script(target.path), timeout=60)
    if rc != 0 or "READY" not in out:
        return SyncResult(target, False, (err.strip().splitlines() or [f"exit {rc}"])[-1])

    env = dict(os.environ, GIT_SSH_COMMAND=f"ssh -F {shlex.quote(str(transport.ssh_config()))}")
    try:
        push = subprocess.run(
            ["git", "-C", repo.root, "push", "-q", "--force", "--no-verify",
             f"--receive-pack={RECEIVE_PACK}",
             f"{host.alias}:{target.path}", f"{commit}:{SYNC_REF}"],
            capture_output=True, text=True, timeout=PUSH_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return SyncResult(target, False, f"git push timed out after {PUSH_TIMEOUT}s")
    if push.returncode != 0:
        msg = (push.stderr.strip().splitlines() or [f"exit {push.returncode}"])[-1]
        return SyncResult(target, False, f"git push failed: {msg}")

    rc, out, err = transport.run(host, _checkout_script(target.path, commit), timeout=120)
    if rc != 0 or "SYNCED" not in out:
        detail = " ".join(err.strip().splitlines()[-3:]) or f"exit {rc}"
        return SyncResult(target, False, f"checkout failed on the server: {detail}")
    return SyncResult(target, True, f"{target.path} now at {repo.describe()}")
