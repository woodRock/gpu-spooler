#!/usr/bin/env bash
# Fake ECS-like cluster on localhost for testing gpu-spooler.
#
#   entry     127.0.0.1:2201   jump hop 1
#   barretts  127.0.0.1:2202   jump hop 2
#   cuda01    127.0.0.1:2211   2 x RTX A5000 (mock nvidia-smi)
#   cuda02    127.0.0.1:2212   4 x A40       (mock nvidia-smi)
#   grid      127.0.0.1:2213   SGE submit host (mock qsub, qstat, qdel)
#
# Every host is an unprivileged sshd running as you. They share one home
# directory ($D/home), like ECS's NFS homes. `up` writes $D/hosts.toml with
# mode = "jump" and a local host.
#
# Usage:
#   dev/fake_cluster.sh up | down | status
#   dev/fake_cluster.sh busy HOST GPU     # pretend another user is on a GPU
#   dev/fake_cluster.sh idle HOST [GPU]   # undo busy
#   dev/fake_cluster.sh stop HOST         # kill a host's sshd and all its connections
#   dev/fake_cluster.sh start HOST
#
#   export GPU_SPOOLER_CONFIG=/tmp/ts-fake-cluster/hosts.toml
set -euo pipefail

D="${TS_FAKE_CLUSTER:-/tmp/ts-fake-cluster}"
HOSTS="entry barretts cuda01 cuda02 grid"
SSHD="${SSHD:-$(command -v sshd || echo /usr/sbin/sshd)}"
ME="$(id -un)"

port_of() {
  case "$1" in
    entry) echo 2201 ;; barretts) echo 2202 ;;
    cuda01) echo 2211 ;; cuda02) echo 2212 ;; grid) echo 2213 ;;
    *) echo "unknown host: $1" >&2; exit 2 ;;
  esac
}

write_mock_nvidia_smi() {
  cat > "$D/bin/nvidia-smi" <<'PY'
#!/usr/bin/env python3
# Mock nvidia-smi: GPUs from $FAKE_CLUSTER/state/$FAKE_HOST.gpus ("name|total_mib"
# per line); a GPU is busy while $FAKE_CLUSTER/busy/$FAKE_HOST.<idx> exists
# (it holds the pid of a dummy process owned by "another user").
import os, sys
D = os.environ.get("FAKE_CLUSTER", "/tmp/ts-fake-cluster")
host = os.environ.get("FAKE_HOST", "")
try:
    gpus = [l.split("|") for l in open(f"{D}/state/{host}.gpus").read().splitlines() if l]
except OSError:
    print("No devices were found"); sys.exit(6)
busy = {}
for i in range(len(gpus)):
    try:
        busy[i] = int(open(f"{D}/busy/{host}.{i}").read().strip())
    except (OSError, ValueError):
        pass
args = sys.argv[1:]
if args[:1] == ["pmon"]:
    print("# gpu         pid   type     sm    mem    enc    dec    jpg    ofa     fb   command")
    print("# Idx           #    C/G      %      %      %      %      %      %     MB   name")
    for i in range(len(gpus)):
        if i in busy:
            print(f"{i:>5} {busy[i]:>11}      C     87     40      -      -      -      -   5000   python")
        else:
            print(f"{i:>5}           -      -      -      -      -      -      -      -      -   -")
elif any(a.startswith("--query-gpu") for a in args):
    for i, (name, total) in enumerate(gpus):
        used = 5000 if i in busy else 1
        print(f"{i}, {name}, {used}, {int(total) - used}, {total}, {87 if i in busy else 0}, Default")
elif any(a.startswith("--query-compute-apps") for a in args):
    for i, pid in sorted(busy.items()):
        print(f"{i}, {pid}")
else:
    for i, (name, total) in enumerate(gpus):
        print(f"GPU {i}: {name} ({'busy' if i in busy else 'idle'}, {total} MiB)")
PY
  chmod +x "$D/bin/nvidia-smi"
}

write_mock_sge() {
  # Like the real thing, the commands are only on PATH after sourcing settings.sh.
  # barretts has an nvidia-smi that fails.
  mkdir -p "$D/sge/bin" "$D/sge/jobs" "$D/sge/nvidia"
  printf '#!/bin/sh\nexit 1\n' > "$D/sge/nvidia/nvidia-smi"
  chmod +x "$D/sge/nvidia/nvidia-smi"
  echo 'export PATH="$FAKE_CLUSTER/sge/bin:$PATH"' > "$D/sge/settings.sh"
  cat > "$D/sge/bin/sge" <<'PY'
#!/usr/bin/env python3
# Mock qsub, qstat and qdel, chosen by the name they're called by. A job is a directory
# $FAKE_CLUSTER/sge/jobs/<id>; it stays in state qw while $FAKE_CLUSTER/sge/hold exists, and
# qstat fails while $FAKE_CLUSTER/sge/down exists. The job script runs as written, except that
# /local/tmp is moved to somewhere an unprivileged user can write.
import os, shutil, signal, subprocess, sys, time
S = os.path.join(os.environ.get("FAKE_CLUSTER", "/tmp/ts-fake-cluster"), "sge")
JOBS = f"{S}/jobs"

def write(path, text):
    with open(path, "w") as f:
        f.write(text)

def qsub(args):
    opts = {}
    while args[0].startswith("-"):
        opts[args[0]] = args[1]
        args = args[2:]
    jid = str(1000 + len(os.listdir(JOBS)) + int(time.time() * 10) % 100000)
    d = f"{JOBS}/{jid}"
    os.mkdir(d)
    write(f"{d}/state", "qw")
    write(f"{d}/name", opts["-N"])
    with open(args[0]) as f:
        write(f"{d}/job.sh", f.read().replace("/local/tmp", f"{S}/local-tmp"))
    p = subprocess.Popen([sys.argv[0], "run", jid, opts["-o"]], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    write(f"{d}/pid", str(p.pid))
    print(f'Your job {jid} ("{opts["-N"]}") has been submitted')

def run(jid, out):
    d = f"{JOBS}/{jid}"
    while os.path.exists(f"{S}/hold"):
        time.sleep(0.2)
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or subprocess.check_output(["id", "-un"], text=True).strip()
    os.makedirs(f"{S}/local-tmp/{user}/{jid}", exist_ok=True)
    write(f"{d}/state", "r")
    with open(out, "ab") as log:
        subprocess.run(["/bin/sh", f"{d}/job.sh"], stdout=log, stderr=log, cwd=os.environ.get("HOME"),
                       env=dict(os.environ, JOB_ID=jid))
    shutil.rmtree(d, ignore_errors=True)

def qstat(args):
    if os.path.exists(f"{S}/down"):
        print("error: commlib error: can't connect to service", file=sys.stderr)
        sys.exit(1)
    if args[:1] == ["-j"]:  # exits 0 even for a job that doesn't exist, like the real one
        return
    print("job-ID  prior   name       user         state submit/start at     queue                          slots ja-task-ID")
    print("-" * 113)
    for jid in sorted(os.listdir(JOBS)):
        try:
            state = open(f"{JOBS}/{jid}/state").read()
            name = open(f"{JOBS}/{jid}/name").read()[:10]
        except OSError:
            continue
        print(f"{jid:>7} 0.55500 {name:<10} {args[-1]:<12} {state:<5} 11/05/2026 12:12:01 all.q@fake.ecs.vuw.ac.nz           1")

def qdel(args):
    d = f"{JOBS}/{args[0]}"
    if not os.path.isdir(d):
        print(f'denied: job "{args[0]}" does not exist', file=sys.stderr)
        sys.exit(1)
    try:
        os.killpg(int(open(f"{d}/pid").read()), signal.SIGKILL)
    except (OSError, ValueError):
        pass
    shutil.rmtree(d, ignore_errors=True)
    print(f"{os.environ.get('USER', 'user')} has deleted job {args[0]}")

name = os.path.basename(sys.argv[0])
if sys.argv[1:2] == ["run"]:
    run(*sys.argv[2:4])
else:
    {"qsub": qsub, "qstat": qstat, "qdel": qdel}[name](sys.argv[1:])
PY
  chmod +x "$D/sge/bin/sge"
  for c in qsub qstat qdel; do ln -sf sge "$D/sge/bin/$c"; done
}

write_hosts_toml() {
  cat > "$D/hosts.toml" <<EOF
# Generated by dev/fake_cluster.sh
[network]
mode = "jump"
probe = "127.0.0.1:2211"

[ssh]
identity_file = "$D/id_ed25519"
control_persist = "120s"
jump_control_persist = "120s"
extra_options = [
  "UserKnownHostsFile=$D/known_hosts",
  "IdentitiesOnly=yes",
  "IdentityAgent=none",
]

[jumps.fake]
chain = ["entry", "barretts"]

[hosts.entry]
hostname = "127.0.0.1"
port = 2201
user = "$ME"

[hosts.barretts]
hostname = "127.0.0.1"
port = 2202
user = "$ME"

[groups.fake]
user = "$ME"
hostname = "127.0.0.1"
jump = "fake"
path_map = [ { local = "$D/local-src", remote = "$D/remote-src" } ]
remote_root = "$D/repos"
hosts = ["cuda01", "cuda02"]

[hosts.cuda01]
port = 2211

[hosts.cuda02]
port = 2212
gpu_type = "A40"

[hosts.grid]
kind = "sge"
hostname = "127.0.0.1"
port = 2213
user = "$ME"
jump = "fake"
remote_root = "$D/repos"
path_map = [ { local = "$D/local-src", remote = "$D/remote-src" } ]
sge_settings = "$D/sge/settings.sh"
sge_wd = "$D/home"

[hosts.local]
kind = "local"
cpu_slots = 2
EOF
}

start_host() {
  local h="$1" port; port="$(port_of "$h")"
  local extra_path=""
  case "$h" in cuda*) extra_path="$D/bin:" ;; grid) extra_path="$D/sge/nvidia:" ;; esac
  cat > "$D/sshd_$h.conf" <<EOF
Port $port
ListenAddress 127.0.0.1
HostKey $D/hostkey
PidFile $D/run/sshd_$h.pid
AuthorizedKeysFile $D/authorized_keys
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM no
StrictModes no
AllowTcpForwarding yes
PermitTTY yes
LogLevel VERBOSE
SetEnv FAKE_HOST=$h FAKE_CLUSTER=$D HOME=$D/home PATH=${extra_path}/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
EOF
  if [ "$(id -u)" = 0 ]; then mkdir -p /run/sshd; fi
  "$SSHD" -f "$D/sshd_$h.conf" -E "$D/run/sshd_$h.log"
}

kill_port() {
  # Kill every process with a TCP socket on this port: the listener, its
  # sessions, and client-side connections (including ControlMasters).
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti "tcp:$1" 2>/dev/null || true)"
  elif command -v fuser >/dev/null 2>&1; then
    pids="$(fuser "$1/tcp" 2>/dev/null || true)"
  fi
  if [ -n "$pids" ]; then kill $pids 2>/dev/null || true; fi
}

stop_host() {
  local h="$1"
  if [ -f "$D/run/sshd_$h.pid" ]; then
    kill "$(cat "$D/run/sshd_$h.pid")" 2>/dev/null || true
    rm -f "$D/run/sshd_$h.pid"
  fi
  kill_port "$(port_of "$h")"
}

cmd_up() {
  mkdir -p "$D/bin" "$D/run" "$D/state" "$D/busy" "$D/home" "$D/local-src" "$D/remote-src" "$D/repos"
  chmod 700 "$D"
  [ -f "$D/hostkey" ] || ssh-keygen -q -t ed25519 -N '' -f "$D/hostkey"
  [ -f "$D/id_ed25519" ] || ssh-keygen -q -t ed25519 -N '' -f "$D/id_ed25519"
  cp "$D/id_ed25519.pub" "$D/authorized_keys"
  chmod 600 "$D/authorized_keys"
  : > "$D/known_hosts"
  printf 'RTX A5000|24564\nRTX A5000|24564\n' > "$D/state/cuda01.gpus"
  printf 'NVIDIA A40|46068\nNVIDIA A40|46068\nNVIDIA A40|46068\nNVIDIA A40|46068\n' > "$D/state/cuda02.gpus"
  write_mock_nvidia_smi
  write_mock_sge
  write_hosts_toml
  for h in $HOSTS; do stop_host "$h"; start_host "$h"; done
  sleep 0.5
  cmd_status
  echo "export GPU_SPOOLER_CONFIG=$D/hosts.toml"
}

cmd_down() {
  for h in $HOSTS; do stop_host "$h"; done
  for f in "$D"/busy/*; do [ -f "$f" ] && cmd_idle_file "$f"; done
  echo "fake cluster stopped"
}

cmd_status() {
  for h in $HOSTS; do
    if [ -f "$D/run/sshd_$h.pid" ] && kill -0 "$(cat "$D/run/sshd_$h.pid")" 2>/dev/null; then
      echo "$h: up (port $(port_of "$h"))"
    else
      echo "$h: down"
    fi
  done
}

cmd_idle_file() {
  kill "$(cat "$1")" 2>/dev/null || true
  rm -f "$1"
}

cmd_busy() {
  local h="$1" gpu="$2" f="$D/busy/$1.$2"
  [ -f "$f" ] && cmd_idle_file "$f"
  nohup sleep 86400 >/dev/null 2>&1 &
  echo $! > "$f"
  echo "$h GPU $gpu marked busy (pid $!)"
}

cmd_idle() {
  local h="$1"
  if [ $# -ge 2 ]; then
    [ -f "$D/busy/$h.$2" ] && cmd_idle_file "$D/busy/$h.$2"
  else
    for f in "$D/busy/$h".*; do [ -f "$f" ] && cmd_idle_file "$f"; done
  fi
  return 0
}

case "${1:-}" in
  up) cmd_up ;;
  down) cmd_down ;;
  status) cmd_status ;;
  busy) [ $# -eq 3 ] || { echo "usage: $0 busy HOST GPU" >&2; exit 2; }; cmd_busy "$2" "$3" ;;
  idle) [ $# -ge 2 ] || { echo "usage: $0 idle HOST [GPU]" >&2; exit 2; }; shift; cmd_idle "$@" ;;
  stop) port_of "$2" >/dev/null; stop_host "$2"; echo "$2 stopped" ;;
  start) port_of "$2" >/dev/null; start_host "$2"; sleep 0.3; echo "$2 started" ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
