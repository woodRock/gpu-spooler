# gpu-spooler

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey?logo=linux&logoColor=white)](https://github.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Servers](https://img.shields.io/badge/GPU%20servers-30-blueviolet?logo=nvidia&logoColor=white)](https://www.ecs.vuw.ac.nz)
[![Status](https://img.shields.io/badge/status-active-brightgreen)](https://github.com)

A distributed GPU task spooler for SSH-reachable GPU servers, set up out of the box for the **VUW ECS GPU cluster**. Queue jobs from an ECS machine or from your laptop (on or off campus), and the background daemon finds a server with free GPUs, starts the job there, and keeps track of it. You don't have to hunt for a free server yourself.

Jobs run **detached** on the server. Closing your laptop, losing wifi or restarting the daemon doesn't kill them. The daemon picks them up again and finishes copying their output.

---

## Contents

- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands](#commands)
  - [Submitting Tasks](#submitting-tasks)
  - [Inspecting Tasks](#inspecting-tasks)
  - [Managing Tasks](#managing-tasks)
  - [GPU Status](#gpu-status)
  - [Cluster Operations](#cluster-operations)
  - [Hosts & Connections](#hosts--connections)
  - [Daemon](#daemon)
- [Configuration](#configuration)
- [Working Off Campus](#working-off-campus)
- [Local Host](#local-host)
- [GPU Servers](#gpu-servers)
- [State & Logs](#state--logs)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## How It Works

```
 You                  Daemon                       GPU server
  │                     │                              │
  ├─ task -G 1 cmd ──►  │                              │
  │  (queued in DB)     │                              │
  │                     ├── ssh → nvidia-smi ───────►  │
  │                     │◄── free GPUs ──────────────  │
  │                     │                              │
  │                     ├── ssh → start detached ───►  │  ~/.task-spooler/jobs/<key>/
  │                     │   CUDA_VISIBLE_DEVICES=0     │    run.sh  pid  out.log  rc
  │                     │                              │
  │                     ├── every 5s: status + new ─►  │
  │                     │◄── log bytes ──────────────  │
  ├─ task -f 1 ───────► │  ~/.task-spooler/logs/1.log  │
```

1. `task -G N cmd` adds a job to a local SQLite queue and auto-starts the daemon.
2. The daemon polls the GPU servers in parallel every 30 seconds with `nvidia-smi`. It polls immediately when a new task is queued.
3. When a server has at least N free GPUs, the daemon starts the command there **in its own session**, in a login shell with `CUDA_VISIBLE_DEVICES` set. Your `~/.bashrc` and conda environments are loaded.
4. Every 5 seconds, one SSH call per server checks each job and copies new output into `~/.task-spooler/logs/<id>.log`. If a server is unreachable, its jobs stay `running` until it comes back.
5. The daemon uses **bin-packing**: it fills servers with fewer GPUs first, so it doesn't fragment GPUs across too many machines.

All SSH traffic goes through a generated config with connection sharing (`ControlMaster`). After the first call, each later one reuses an open connection and takes milliseconds.

---

## Installation

**Requirements:** Python 3.10+, and SSH access to the GPU servers with key-based auth. Off campus, you can use a key or a one-time password on the jump host (see [Working Off Campus](#working-off-campus)).

```bash
git clone <repo-url> task-spooler
cd task-spooler
bash install.sh
```

The installer creates a `.venv` in the project directory, installs the package in editable mode, and symlinks `task` and `gpu-status` into `~/.local/bin`.

If `~/.local/bin` is not in your `PATH`, add this to your `~/.bashrc` or `~/.zshrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Since the install is **editable**, any changes you make to the source files take effect immediately without reinstalling.

### Installing packages across the cluster

To install a Python package on every GPU server at once (e.g. after adding a new dependency):

```bash
task -R "cd /vol/ecrg-solar/woodj4/depth-learning && pip install -e ."
```

---

## Quick Start

```bash
# Check which servers have free GPUs
gpu-status

# Queue a training job requiring 1 GPU
task -G 1 python3 train.py --epochs 50

# Queue a job requiring 2 GPUs with a label
task -G 2 -n "ablation run" python3 train.py --model large

# Watch the queue
task -l

# Follow live output
task -f 1

# Kill everything and reset
task -K
```

On a laptop, run `task --init-config` once and then `task --hosts` (see [Configuration](#configuration)).

---

## Commands

### Submitting Tasks

```
task -G <N> [options] <command...>
```

| Flag | Description |
|------|-------------|
| `-G N` / `--gpus N` | **Required.** Number of GPUs the job needs. `-G 0` is a CPU-only job, which runs on hosts with `cpu_slots` (see [Local Host](#local-host)). |
| `-m GB` / `--min-mem GB` | Only use servers with at least this much VRAM per GPU, e.g. `-m 40` for the 48 GB cards. |
| `-H SPEC` / `--host SPEC` | Only run on these hosts, groups or tags (comma separated), e.g. `-H cuda14,cuda15` or `-H local`. |
| `-n LABEL` / `--name LABEL` | Optional human-readable label shown in the task list. |
| `-d PATH` / `--dir PATH` | Working directory (default: current directory). It is translated with the host's `path_map`. Without a matching entry, the path is used unchanged on the server. |

**Examples:**

```bash
task -G 1 python3 train.py --task counting --dataset easy --epochs 100
task -G 2 -n "multi-gpu run" bash experiments/run.sh
task -G 1 -d /vol/ecrg-solar/woodj4/project python3 eval.py
task -G 1 -m 40 -H cuda14,cuda15 python3 big_model.py
task -G 0 -H local 'python3 make_plots.py && echo done'
```

> Arguments after the command, including flags like `--task` and `--dataset`, are passed through verbatim and never interpreted by the spooler. A **single** quoted argument is run as shell code, so `task -G 1 'cd sub && ./run.sh'` works.

---

### Inspecting Tasks

| Command | Description |
|---------|-------------|
| `task -l` / `task --list` | List tasks: status, `-H` constraint (HOST), server, GPU indices, runtime and label. The footer shows the network mode. |
| `task -i ID` / `task --info ID` | Detailed info for a single task: server, GPU type, VRAM, working dir, timestamps, exit code, log path. |
| `task -o ID` / `task --output ID` | Print the full saved output (stdout + stderr) of a task. |
| `task -f ID` / `task --follow ID` | Stream live output as the task runs (`tail -f` style). Output arrives in roughly 5-second batches. Waits if the task hasn't started yet. |

**Task status colours:**

| Status | Meaning |
|--------|---------|
| `queued` | Waiting for a free GPU |
| `running` | Currently executing on a server |
| `done` | Finished successfully (exit 0) |
| `failed` | Finished with a non-zero exit code (after 3 automatic retries) |
| `killed` | Manually killed via `task -k` or `task -K` |
| `cancelled` | Cancelled before it started |
| `lost` | The job disappeared from its server without recording an exit code (e.g. the server rebooted), or it was started by an older version of the spooler. Not retried automatically. |

---

### Managing Tasks

| Command | Description |
|---------|-------------|
| `task -k ID` / `task --kill ID` | Kill a running task or cancel a queued one. Sends SIGTERM to the job's whole process group on the server, then SIGKILL after 10 s if it is still alive. If the server can't be reached, the task is still marked `killed`, and you get a warning with the command to check it yourself. |
| `task -K` / `task --kill-all` | Kill all running tasks (in parallel), cancel all queued tasks, stop the daemon, clear the task list, and reset the ID counter to 1. |
| `task -c` / `task --clear` | Remove finished tasks (done/failed/killed/cancelled/lost) from the list and delete their logs and server-side job directories. |
| `task -w ID` / `task --wait ID` | Block until the specified task finishes. Useful in scripts. |

---

### GPU Status

```bash
gpu-status                  # per-server summary table
gpu-status --detail         # one row per GPU
gpu-status --free-only      # hide fully-busy servers
gpu-status --server cuda14  # check one server (or a group/tag, comma separated)
gpu-status --debug          # print raw SSH output (for diagnosing issues)
```

You can also check GPU status from within the task command:

```bash
task --gpu-status
```

**Example output:**

```
SERVER          GPU TYPE        GPUs  FREE   VRAM FREE   VRAM TOTAL  USERS / PROCESSES
──────────────────────────────────────────────────────────────────────────────────────
cuda14          A40                4     2     96.0 GB    192.0 GB   woodj4: python3 train.py
cuda20          L40S               4     4    192.0 GB    192.0 GB   idle
cuda16          RTX A6000          3     0      0.0 GB    144.0 GB   jsmith: bash run.sh
...
```

---

### Cluster Operations

Run any shell command on **every configured server in parallel**:

```bash
task -R "CMD"
```

Results stream in as servers respond, with a live progress indicator. Failed servers print their full output at the end.

**Examples:**

```bash
# Install a package everywhere
task -R "cd /vol/ecrg-solar/woodj4/depth-learning && pip install -e ."

# Check Python version on all servers
task -R "python3 --version"

# Check disk usage on the shared volume
task -R "df -h /vol/ecrg-solar"
```

---

### Hosts & Connections

| Command | Description |
|---------|-------------|
| `task --hosts` | Table of configured hosts: kind, group, route (direct or via jump hosts), live reachability, GPU count, CPU slots and path mappings. Also shows the network mode and where the config came from. |
| `task --connect` | Log in to each jump host interactively, once. Use this when the jump host wants a password or OTP; the connection then stays open for the daemon. |
| `task --ssh HOST` | Open an interactive shell on a host through the same route the daemon uses. Useful for debugging. |
| `task --init-config` | Write a commented `~/.task-spooler/hosts.toml` template. Never overwrites an existing file. |

---

### Daemon

The daemon starts automatically when you submit a job. You rarely need to manage it directly.

| Command | Description |
|---------|-------------|
| `task --daemon` | Start the daemon if it isn't already running. |
| `task --stop` | Stop the daemon. Running tasks keep running on their servers. The next daemon picks them up again and copies the output it missed. |

Daemon logs are written to `~/.task-spooler/daemon.log`.

---

## Configuration

Hosts are read from `~/.task-spooler/hosts.toml`. You can point at another file with the `TASK_SPOOLER_CONFIG` environment variable. **Without the file, the built-in ECS server list is used** with plain `ssh <name>`, exactly as before. That's all you need on an ECS machine.

Create a starting point with:

```bash
task --init-config        # writes ~/.task-spooler/hosts.toml
task --hosts              # check what it resolves to
```

The template contains every ECS server. The main parts are:

```toml
[network]
mode = "auto"                  # auto | direct | jump
# probe = "cuda00.ecs.vuw.ac.nz"  # host used to detect if you're on campus

[ssh]
# identity_file = "~/.ssh/id_ed25519"
control_persist = "10m"        # how long server connections stay open
jump_control_persist = "12h"   # how long jump-host connections stay open
extra_options = []             # raw "Key=Value" ssh options

[jumps.ecs]
chain = ["woodj4@entry.ecs.vuw.ac.nz", "woodj4@barretts.ecs.vuw.ac.nz"]

[groups.ecs]
user = "woodj4"
domain = "ecs.vuw.ac.nz"       # hostname = "<name>.<domain>"
jump = "ecs"
path_map = [ { local = "~/code", remote = "/vol/ecrg-solar/woodj4" } ]
hosts = ["cuda-small0", "cuda00", "cuda01", "...", "bordeaux"]

[hosts.cuda14]                 # per-host settings override the group
gpu_type = "A40"
gpu_mem_gb = 48                # used by -m/--min-mem (else detected)
enabled = true                 # false = never dispatch here

[hosts.local]
kind = "local"                 # run on this machine, no ssh
cpu_slots = 2
```

- **Groups** give defaults to their hosts. A host's group names also act as tags, which you can use with `-H`, `gpu-status --server`, and so on. You can add more with `tags = [...]`.
- **`path_map`** translates the working directory when a job runs on that host, using the longest matching prefix. For example, `~/code/proj` on your laptop becomes `/vol/ecrg-solar/woodj4/proj` on ECS. The spooler doesn't copy code. It only translates the path, so the code needs to exist at the remote path.
- **Jump chain** entries are either `user@host[:port]` or the name of a `[hosts.X]` entry.
- From the config, the spooler generates `~/.task-spooler/ssh_config.direct` and `ssh_config.jump`, and calls `ssh -F <file>`. Your own `~/.ssh/config` is included last, so settings such as keys and agents still apply.

---

## Working Off Campus

Since Trimester 2 2025, off-campus SSH to ECS must go through `entry.ecs.vuw.ac.nz`. The route is **laptop → entry → barretts → cudaXX**. With `mode = "auto"`, the spooler checks whether it can reach the servers directly (on campus, including VUW wifi and the halls). If it can't, it goes through the jump chain. The result is cached for 5 minutes, and the daemon re-checks after that. `task --hosts`, `task -l` and `gpu-status` show the mode in use.

Each hop keeps one shared connection (with `ControlPersist`), so the 30 servers don't each log in to `entry` separately.

- **With SSH keys** on entry (and your key on barretts and the servers), nothing else is needed.
- **With OTP + password**, run `task --connect` once and type your password and code. The connection to entry stays open for `jump_control_persist` (12 h by default), and the daemon reuses it. When the daemon reports `authentication failed`, run `task --connect` again.

Jobs run detached on the server. If your laptop sleeps or changes network, the jobs keep running, and the daemon catches up on their output when it can reach the servers again.

---

## Local Host

Add a `kind = "local"` host to run jobs on the machine the daemon runs on:

```toml
[hosts.local]
kind = "local"
cpu_slots = 2      # how many -G 0 jobs may run here at once
```

If the machine has NVIDIA GPUs, `-G N` jobs can use them too. Without `nvidia-smi`, it only takes CPU jobs. `-G 0` jobs go only to hosts with free `cpu_slots`. Use `-H local` to pin a job to this machine.

---

## GPU Servers

| Server | GPUs | Type | VRAM/GPU | CPU Cores | RAM |
|--------|-----:|------|----------:|----------:|----:|
| cuda-small0 | 4 | RTX A4000 | 16 GB | 16 | 192 GB |
| cuda-small1 | 3 | RTX 4000 Ada | 20 GB | 16 | 144 GB |
| cuda00 | 2 | RTX 6000 | 24 GB | 32 | 96 GB |
| cuda01–04 | 2 | RTX A5000 | 24 GB | 16 | 64 GB |
| cuda05–08 | 3 | RTX A5000 | 24 GB | 24 | 144 GB |
| cuda09 | 3 | RTX 6000 | 24 GB | 32 | 96 GB |
| cuda10–11 | 3 | RTX A5000 | 24 GB | 24 | 112–128 GB |
| cuda12–13 | 3 | RTX 6000 | 24 GB | 32 | 96 GB |
| cuda14–15 | 4 | A40 | 48 GB | 64 | 256 GB |
| cuda16 | 3 | RTX A6000 | 48 GB | 32 | 192 GB |
| cuda17 | 3 | RTX 6000 | 24 GB | 24 | 144 GB |
| cuda18 | 3 | RTX A6000 | 48 GB | 32 | 192 GB |
| cuda19 | 4 | L4 | 24 GB | 64 | 256 GB |
| cuda20 | 4 | L40S | 48 GB | 64 | 384 GB |
| cuda21–22 | 2 | RTX 6000 Ada | 48 GB | 48 | 256 GB |
| gryphon | 1 | RTX A6000 | 48 GB | 64 | 192 GB |
| red-tomatoes | 2 | RTX A6000 | 48 GB | 32 | 96 GB |
| piccolo | 2 | RTX A5000 | 24 GB | 20 | 128 GB |
| the-villa | 1 | RTX A6000 | 48 GB | 64 | 192 GB |
| bordeaux | 1 | RTX 6000 | 24 GB | 20 | 64 GB |

**Total: 30 servers · 80 GPUs**

---

## State & Logs

All state is stored in `~/.task-spooler/`:

```
~/.task-spooler/
├── hosts.toml          # your host configuration (optional)
├── tasks.db            # SQLite database (all task records)
├── daemon.pid          # PID of the running daemon
├── daemon.log          # Daemon stdout/stderr
├── ssh_config.direct   # generated ssh configs (don't edit)
├── ssh_config.jump
├── cm/                 # shared ssh connection sockets
├── network.json        # cached network-mode probe
├── gpu_snapshot.json   # latest GPU query from the daemon
└── logs/
    ├── 1.log           # Output for task 1
    └── ...
```

On each server, a job lives in `~/.task-spooler/jobs/<id>-<random>/` (`run.sh`, `pid`, `out.log`, `rc`). ECS home directories are shared over NFS, so this is the same directory on every ECS machine. `task -c` removes these directories for finished tasks.

Task logs include a header with the command, server, GPU indices, start time, and working directory, followed by the raw stdout/stderr of your job.

---

## Troubleshooting

**Command not found on the remote server**

The daemon sources `~/.bashrc` before running your command, which should load conda environments and `~/.local/bin`. If a command is still missing, verify it's accessible in a non-interactive SSH session:

```bash
task --ssh cuda14       # then: which mycommand
```

**Working directory not found (`No such file or directory`)**

NFS automount paths (e.g. `/am/...`) may not be mounted yet when the SSH session starts. The spooler runs `ls <dir>` first to trigger the automount. If this still fails, use `-d` to provide the path as seen from the remote server:

```bash
task -G 1 -d /vol/ecrg-solar/woodj4/project python3 train.py
```

**All servers show as unreachable**

Run `task --hosts` to see the network mode and the SSH error for each host. SSH key-based auth must be set up for the servers. If the error mentions authentication and you're off campus, run `task --connect`. To test one host by hand:

```bash
ssh -F ~/.task-spooler/ssh_config.direct cuda01 echo ok    # or ssh_config.jump
```

If this prompts for a password, add your public key to the servers:

```bash
ssh-copy-id cuda01
```

**GPU shows as busy but no jobs are running**

`gpu-status` uses `nvidia-smi pmon` which shows all processes using the GPU, including display servers and idle CUDA contexts. Run `gpu-status --detail` to see the specific process names and users holding each GPU.

**Daemon disappeared while a task was running**

Start it again with `task --daemon`. Running jobs are picked up again, and the output they wrote in the meantime is copied over. Only tasks started by the old, pre-detached version of the spooler are marked `lost`.

**`task -k` couldn't reach the server**

The task is marked `killed`, but the process may still be running. Check and kill it by hand with the command printed in the warning (`task --ssh <server>`, then `kill -- -<PGID>`). `task -i ID` shows the PGID.

---

## Development

Tests use pytest. The integration tests start a fake cluster on localhost: four unprivileged `sshd`s (entry, barretts, cuda01 with 2 × A5000, cuda02 with 4 × A40) with a mock `nvidia-smi`. They are skipped if `sshd` isn't available.

```bash
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest tests

# Use the fake cluster by hand
dev/fake_cluster.sh up
export TASK_SPOOLER_CONFIG=/tmp/ts-fake-cluster/hosts.toml
export HOME=$(mktemp -d /tmp/tsh.XXXX)       # optional: isolate ~/.task-spooler
dev/fake_cluster.sh busy cuda01 0            # another "user" takes GPU 0
dev/fake_cluster.sh stop barretts            # simulate a dropped jump host
dev/fake_cluster.sh start barretts
dev/fake_cluster.sh down
```

The sshd logs in `/tmp/ts-fake-cluster/run/sshd_*.log` show the traffic through each hop.
