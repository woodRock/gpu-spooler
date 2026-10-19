"""
Detached jobs.

Each launch attempt gets a job directory on the host that runs it,
$HOME/.gpu-spooler/jobs/<job_key>/, holding run.sh, pid, out.log and (once
the command exits) rc. Jobs are started in their own session, so they keep
running when the ssh connection that launched them goes away; the daemon
polls the directory to sync the log and pick up the exit code.
"""

import base64
import secrets
import shlex
from dataclasses import dataclass
from typing import Optional

from gpu_spooler import provenance
from gpu_spooler.repo import FIND_GIT, sh_path

JOBS_DIR = "$HOME/.gpu-spooler/jobs"
CHUNK_BYTES = 1048576        # max log bytes fetched per job per poll


def new_job_key(task_id: int) -> str:
    return f"{task_id}-{secrets.token_hex(3)}"


def launch_script(key: str, wdir: str, gpus: str, command: str, delay: bool,
                  probe: Optional[str] = None) -> str:
    """Script (for `bash -s`) that starts a job detached and prints `TS_PID <pid>`.

    probe: when the task is recorded with taura, how to run taura's environment probe like the job's
    command (e.g. `PYTHONPATH=src python3`); the job then prints a `TAURA_ENV {...}` line first."""
    q_wdir = sh_path(wdir)
    eof = f"TS_EOF_{key.replace('-', '_')}"
    probe_file = probe_run = ""
    if probe:
        peof = f"TS_PROBE_{key.replace('-', '_')}"
        probe_file = (f'cat > "$D/taura_probe.py" <<\'{peof}\'\n' + provenance.PROBE.strip("\n")
                      + f"\n{peof}\n")
        probe_run = f'{probe} "$D/taura_probe.py" 2>/dev/null | sed "s/^/{provenance.ENV_MARK}/" || true\n'
    return f'''\
D="{JOBS_DIR}/{key}"
mkdir -p "$D" || exit 1
{probe_file}cat > "$D/run.sh" <<'{eof}'
D="{JOBS_DIR}/{key}"
source ~/.bashrc 2>/dev/null || true
ls {q_wdir} > /dev/null 2>&1
cd {q_wdir} || {{ echo "gpu-spooler: cannot cd to "{q_wdir} >&2; echo 1 > "$D/rc.tmp"; mv "$D/rc.tmp" "$D/rc"; exit 1; }}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES={shlex.quote(gpus)}
{"sleep 2" if delay else ":"}
{probe_run}( {command}
)
rc=$?
echo "$rc" > "$D/rc.tmp" && mv "$D/rc.tmp" "$D/rc"
exit $rc
{eof}
ts_detach() {{
  trap '' HUP     # like nohup: an ignored SIGHUP survives exec (macOS nohup
                  # refuses to run outside a console session)
  if command -v setsid >/dev/null 2>&1; then exec setsid "$@"
  else exec python3 -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' "$@"; fi
}}
cd "$HOME"
# The commit the job will run, for run provenance (taura): HEAD and uncommitted changes.
# git may only be on a login shell's PATH (on ECS it's in /usr/pkg/bin).
{FIND_GIT}( cd {q_wdir} 2>/dev/null && h=$(git rev-parse -q --verify HEAD 2>/dev/null) \
  && echo "TS_GIT $h $(git status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')" )
ts_detach bash -l "$D/run.sh" > "$D/out.log" 2>&1 < /dev/null &
echo $! > "$D/pid"
echo "TS_PID $!"
'''


def parse_launch(stdout: str) -> Optional[int]:
    for line in stdout.splitlines():
        if line.startswith("TS_PID "):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def parse_git(stdout: str) -> Optional[tuple[str, int]]:
    """(commit, uncommitted changes) the launch script saw in the working directory, if a git repo."""
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "TS_GIT":
            try:
                return parts[1], int(parts[2])
            except ValueError:
                return None
    return None


def status_script(specs: list[tuple[str, int]]) -> str:
    """Script reporting state and new log bytes for (job_key, offset) pairs."""
    items = " ".join(shlex.quote(f"{k}:{off}") for k, off in specs)
    return f'''\
for spec in {items}; do
  k=${{spec%%:*}}; off=${{spec##*:}}; D="{JOBS_DIR}/$k"
  printf '@@JOB %s\\n' "$k"
  if [ -f "$D/rc" ]; then printf 'STATE done %s\\n' "$(cat "$D/rc")"
  elif [ ! -d "$D" ]; then echo 'STATE missing'
  elif [ -f "$D/pid" ] && kill -0 "$(cat "$D/pid")" 2>/dev/null; then echo 'STATE running'
  elif [ -f "$D/rc" ]; then printf 'STATE done %s\\n' "$(cat "$D/rc")"
  else echo 'STATE gone'; fi
  printf 'SIZE %s\\n' "$(wc -c < "$D/out.log" 2>/dev/null | tr -d ' ' || echo 0)"
  printf 'DATA '; tail -c +$((off+1)) "$D/out.log" 2>/dev/null | head -c {CHUNK_BYTES} | base64 | tr -d '\\n'; echo
done
echo '@@END'
'''


@dataclass
class JobStatus:
    state: str                 # running | done | missing | gone
    rc: Optional[int] = None
    size: int = 0
    data: bytes = b""


def parse_status(stdout: str) -> Optional[dict[str, JobStatus]]:
    """Parse status_script output; None if it was cut short."""
    out: dict[str, JobStatus] = {}
    cur: Optional[JobStatus] = None
    complete = False
    for line in stdout.splitlines():
        if line.startswith("@@JOB "):
            cur = out[line[6:].strip()] = JobStatus(state="gone")
        elif line == "@@END":
            complete = True
            break
        elif cur is None:
            continue
        elif line.startswith("STATE "):
            parts = line.split()
            cur.state = parts[1]
            if cur.state == "done":
                try:
                    cur.rc = int(parts[2])
                except (IndexError, ValueError):
                    cur.rc = 1
        elif line.startswith("SIZE "):
            try:
                cur.size = int(line[5:].strip() or 0)
            except ValueError:
                cur.size = 0
        elif line.startswith("DATA"):
            try:
                cur.data = base64.b64decode(line[5:].strip())
            except ValueError:
                return None
    return out if complete else None


def kill_script(key: str, grace: int = 10) -> str:
    """
    SIGTERM the job's process group, then SIGKILL it after `grace` seconds if
    it is still alive (in a detached background step, so this returns at once).
    Prints KILLED, NOT_RUNNING or NO_JOB.
    """
    return f'''\
D="{JOBS_DIR}/{key}"
[ -f "$D/pid" ] || {{ echo NO_JOB; exit 0; }}
P=$(cat "$D/pid")
if kill -TERM -- -"$P" 2>/dev/null; then
  echo KILLED
  ( trap '' HUP; i=0
    while [ $i -lt {grace} ]; do sleep 1; kill -0 -- -"$P" 2>/dev/null || exit 0; i=$((i+1)); done
    kill -KILL -- -"$P" 2>/dev/null ) > /dev/null 2>&1 < /dev/null &
else
  echo NOT_RUNNING
fi
'''


def cleanup_script(task_ids: list[int]) -> str:
    """Remove job directories for every launch attempt of the given tasks."""
    globs = " ".join(f'"$J"/{int(t)}-*' for t in task_ids)
    return f'J="{JOBS_DIR}"\nrm -rf {globs}\n'
