# gpu-spooler

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey?logo=linux&logoColor=white)](https://github.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Servers](https://img.shields.io/badge/GPU%20servers-30-blueviolet?logo=nvidia&logoColor=white)](https://www.ecs.vuw.ac.nz)
[![Status](https://img.shields.io/badge/status-active-brightgreen)](https://github.com)

A distributed GPU task spooler for SSH-reachable GPU servers, set up out of the box for the **VUW ECS GPU cluster**. Queue jobs from an ECS machine or from your laptop (on or off campus), and the background daemon finds a server with free GPUs, starts the job there, and keeps track of it. You don't have to hunt for a free server yourself.

Jobs run **detached** on the server. Closing your laptop, losing wifi or restarting the daemon doesn't kill them. The daemon picks them up again and finishes copying their output.

---

## Quick Start

```bash
curl -fsSL https://raw.githubusercontent.com/woodRock/gpu-spooler/main/install.sh | bash
```

This installs `task`, `gpu-status` and `tui` into `~/.local/bin` (see [Installation](#installation)).

On an ECS machine that's all the setup. On a laptop, once:

```bash
task config init --user woodj4   # writes ~/.gpu-spooler/hosts.toml with your ECS username
task hosts                       # can every server be reached?
task connect                     # only if the jump host asks for a password / OTP
```

Then:

```bash
task gpus                                      # which servers have free GPUs
task submit -- python3 train.py --epochs 50    # 1 GPU unless -g says otherwise
task list                                      # the queue
task logs -f 1                                 # follow task 1's output
task kill 1                                    # cancel or kill it
```

From a laptop checkout of a git repo, push the code first and copy the results back after:

```bash
task sync                                      # copy this repo's working tree to the servers
id=$(task submit -q -- python3 train.py)       # -q prints only the task ID
task fetch results --after "$id"               # copy results/ back once the task is done
```

`task -h` lists every command and `task COMMAND -h` its options, with examples. The same short flag means the same thing in every command:

| Flag | Meaning |
|------|---------|
| `-H SPEC` / `--hosts SPEC` | hosts, groups or tags (comma separated) |
| `-C DIR` / `--dir DIR` | the directory to act on, as with `git -C` |
| `-g N` / `--gpus N` | GPUs a task needs |
| `-t MODEL` / `--gpu-type MODEL` | GPU model, as `nvidia-smi` names it |
| `-n LABEL` / `--name LABEL` | a task's label |
| `-q` / `--quiet` | print only the task ID |
| `-f` | follow (`logs -f`), or only free GPUs (`gpus -f`) |

---

## Contents

- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Commands](#commands)
  - [Submitting Tasks](#submitting-tasks)
  - [Inspecting Tasks](#inspecting-tasks)
  - [Managing Tasks](#managing-tasks)
  - [Interactive UI (`tui`)](#interactive-ui-tui)
  - [GPU Status](#gpu-status)
  - [Cluster Operations](#cluster-operations)
  - [Hosts & Connections](#hosts--connections)
  - [Daemon](#daemon)
- [Configuration](#configuration)
- [Working Off Campus](#working-off-campus)
- [Running Your Git Repos from a Laptop](#running-your-git-repos-from-a-laptop)
  - [Fetching Results](#fetching-results)
  - [Run Provenance (taura)](#run-provenance-taura)
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
  ├─ task submit cmd ►  │                              │
  │  (queued in DB)     │                              │
  │                     ├── ssh → nvidia-smi ───────►  │
  │                     │◄── free GPUs ──────────────  │
  │                     │                              │
  │                     ├── ssh → start detached ───►  │  ~/.gpu-spooler/jobs/<key>/
  │                     │   CUDA_VISIBLE_DEVICES=0     │    run.sh  pid  out.log  rc
  │                     │                              │
  │                     ├── every 5s: status + new ─►  │
  │                     │◄── log bytes ──────────────  │
  ├─ task logs -f 1 ──► │  ~/.gpu-spooler/logs/1.log  │
```

1. `task submit -g N cmd` adds a job to a local SQLite queue and auto-starts the daemon.
2. The daemon polls the GPU servers in parallel every 30 seconds with `nvidia-smi`. It polls immediately when a new task is queued.
3. When a server has at least N free GPUs, the daemon starts the command there **in its own session**, in a login shell with `CUDA_VISIBLE_DEVICES` set. Your `~/.bashrc` and conda environments are loaded.
4. Every 5 seconds, one SSH call per server checks each job and copies new output into `~/.gpu-spooler/logs/<id>.log`. If a server is unreachable, its jobs stay `running` until it comes back.
5. The daemon uses **bin-packing**: it fills servers with fewer GPUs first, so it doesn't fragment GPUs across too many machines.

All SSH traffic goes through a generated config with connection sharing (`ControlMaster`). After the first call, each later one reuses an open connection and takes milliseconds.

---

## Installation

**Requirements:** Python 3.10+, and SSH access to the GPU servers with key-based auth. Off campus, you can use a key or a one-time password on the jump host (see [Working Off Campus](#working-off-campus)).

```sh
curl -fsSL https://raw.githubusercontent.com/woodRock/gpu-spooler/main/install.sh | bash
```

This installs `task`, `gpu-status` and `tui` with [uv](https://docs.astral.sh/uv/) as an isolated tool in `~/.local/bin`,
installing uv first if needed (and Python 3.11 if the machine doesn't have it). No sudo. Add
`bash -s -- --upgrade` to update to the latest `main`, or `bash -s -- --uninstall` to remove it:

```sh
curl -fsSL https://raw.githubusercontent.com/woodRock/gpu-spooler/main/install.sh | bash -s -- --upgrade
```

For development, from a checkout: `bash install.sh` creates a `.venv` in the checkout, installs the package in editable mode (with the optional [Textual](https://textual.textualize.io) dependency for `tui`), and symlinks the three commands into `~/.local/bin` (`dev/install-dev.sh`). The daemon then runs your checkout's code.


There are three commands:

- **`task`** is the non-interactive CLI. It's the one to use from scripts and tools such as Claude Code. It only needs the standard library.
- **`tui`** is an interactive view of the queue, logs and GPUs.
- **`gpu-status`** prints a one-off GPU availability table.

`task` and `tui` are thin front ends over the same code (`gpu_spooler/core.py`), so they always behave the same way.

If `~/.local/bin` is not in your `PATH`, add this to your `~/.bashrc` or `~/.zshrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Since the install is **editable**, any changes you make to the source files take effect immediately without reinstalling.

To install a Python package on every server at once, see [Cluster Operations](#cluster-operations).

---

## Commands

### Submitting Tasks

```
task submit [-g N] [-H SPEC] [-t MODEL] [-m GB] [-n LABEL] [-C DIR] [-q] -- COMMAND...
```

| Flag | Description |
|------|-------------|
| `-g N` / `--gpus N` | GPUs the job needs (default 1). `-g 0` is a CPU-only job, which runs on hosts with `cpu_slots` (see [Local Host](#local-host)). |
| `-H SPEC` / `--hosts SPEC` | Only run on these hosts, groups or tags (comma separated), e.g. `-H cuda14,cuda15` or `-H local`. |
| `-t MODEL` / `--gpu-type MODEL` | Only use GPUs that `nvidia-smi` names this model, e.g. `-t "RTX A5000"`; comma separate several. Exact match, ignoring case and a leading `NVIDIA `, so `RTX 6000` does not also match `RTX 6000 Ada Generation`. Use it when results must come from one GPU model (seeded CUDA runs are bit-reproducible only on the same model). |
| `-m GB` / `--min-mem GB` | Only use GPUs with at least this much memory, e.g. `-m 40` for the 48 GB cards. |
| `-n LABEL` / `--name LABEL` | A label shown in the task list. |
| `-C DIR` / `--dir DIR` | Working directory (default: the current one). On the server it becomes the `path_map` translation, or the git repo's copy under `remote_root` (see [Running Your Git Repos from a Laptop](#running-your-git-repos-from-a-laptop)). Otherwise the path is used unchanged. |
| `-q` / `--quiet` | Print only the task ID (warnings go to stderr), for scripts: `id=$(task submit -q -- ./run.sh)`. |

**Examples:**

```bash
task submit -- python3 train.py --task counting --dataset easy --epochs 100
task submit -g 2 -n "multi-gpu run" -- bash experiments/run.sh
task submit -C /vol/ecrg-solar/woodj4/project -- python3 eval.py
task submit -m 40 -H cuda14,cuda15 -- python3 big_model.py
task submit -g 0 -H local -- 'python3 make_plots.py && echo done'
```

> Everything after `--` is the command, passed through verbatim (the `--` can be left out when the command doesn't start with `-`). A **single** quoted argument is run as shell code, so `task submit -- 'cd sub && ./run.sh'` works.

---

### Inspecting Tasks

| Command | Description |
|---------|-------------|
| `task list` / `task ls` | List tasks: status, `-H` constraint (HOST), server, GPU indices, runtime and label. The footer shows the network mode and, with `[limits] max_gpus`, the GPUs in use. `-a` / `--all` lists every task, not only the 200 most recent. |
| `task info ID` | Details of one task: server and GPU model, working dir, timestamps, exit code, log path. |
| `task logs ID` | Print the task's saved output (stdout + stderr). |
| `task logs -f ID` | Follow the output as the task runs (`tail -f` style; it arrives in roughly 5-second batches). Waits if the task hasn't started yet. |

**Task status colours:**

| Status | Meaning |
|--------|---------|
| `queued` | Waiting for a free GPU |
| `running` | Currently executing on a server |
| `done` | Finished successfully (exit 0) |
| `failed` | Finished with a non-zero exit code (after 3 automatic retries) |
| `killed` | Killed with `task kill` |
| `cancelled` | Cancelled before it started |
| `lost` | The job disappeared from its server without recording an exit code (e.g. the server rebooted), or it was started by an older version of the spooler. Not retried automatically. |

---

### Managing Tasks

| Command | Description |
|---------|-------------|
| `task kill ID...` | Kill running tasks or cancel queued ones. Sends SIGTERM to each job's whole process group on its server, then SIGKILL after 10 s if it is still alive. If a server can't be reached, the task is still marked `killed`, and you get a warning with the command to check it yourself. |
| `task kill --all` | The same for every queued and running task, in parallel. |
| `task wait ID... [--timeout S]` | Block until the tasks finish, reading only the local database (no SSH). The exit code says how they ended (table below). Prints one line per task to stderr, e.g. `task 12: done (exit 0, 3m 41s)`, and nothing to stdout. Never prompts. |
| `task resubmit ID` | Queue a copy of a task with the same command, GPUs, GPU type, memory, hosts, label and directory (`-q` prints only the new ID). |
| `task clear` | Remove finished tasks (done/failed/killed/cancelled/lost) from the list and delete their logs and server-side job directories. |
| `task clear --reset` | Also forget everything else and restart IDs at 1. Refused while tasks are queued or running (`task kill --all` first). |

**`task wait` exit codes**, for scripts and agents (`task wait "$id"; rc=$?`):

| Final state | Exit code |
|-------------|-----------|
| `done` | 0 |
| `failed` | the job's own exit code (1 if it was 0 or unknown) |
| `killed`, `cancelled` | 130 |
| `lost` | 75 |
| `--timeout` reached first | 124 (the task keeps running) |
| unknown task ID | 1 |
| Ctrl+C while waiting | 130 (the task keeps running) |

`queued` and `running` keep waiting, and that includes a task re-queued for an automatic retry. A task that has already finished returns at once. With several IDs, `wait` waits for all of them. It exits 0 only if every one is `done`; otherwise it returns the first non-zero code in ID order.

---

### Interactive UI (`tui`)

```bash
tui            # or: task tui
```

The window shows a live queue table (newest first). Below it are tabs for the selected task's **log**, a **GPU** heatmap, and task **info**.

| Key | Action |
|-----|--------|
| `↑` `↓` | Select a task |
| `k` | Kill a running task or cancel a queued one (asks first) |
| `r` | Resubmit the selected task (same as `task resubmit`) |
| `n` | New task: a form with the same fields as `task submit -g/-m/-H/-n/-C` |
| `s` | Sync the git repo `tui` was started in (same as `task sync`) |
| `f` | Follow: show the log full screen (press `f` again to return) |
| `l` / `g` / `i` | Switch to the log / GPUs / info tab |
| `c` | Clear finished tasks (same as `task clear`, asks first) |
| `d` | Start the daemon if it is stopped |
| `q` | Quit |

The GPU heatmap has one row per server and one cell per GPU. A cell is **green** when the GPU is free, **blue** when one of your tasks is using it (the cell shows the task ID), and **red** when it's in use by someone else (the cell shows their username). `tui` doesn't connect to the servers to get this. It reads `~/.gpu-spooler/gpu_snapshot.json`, which the daemon refreshes every 30 s while a `tui` is open. So the daemon needs to be running (press `d` to start it).

If Textual isn't installed, `tui` prints how to install it: `pip install -e '.[tui]'`, or re-run `install.sh`.

---

### GPU Status

```bash
task gpus                   # per-server summary table (gpu-status is the same command)
task gpus -d                # --detail: one row per GPU
task gpus -f                # --free: hide fully-busy servers
task gpus -H cuda14         # one server (or a group/tag, comma separated)
task gpus --debug           # print raw SSH output (for diagnosing issues)
```

`GPU TYPE` is what `nvidia-smi` reports, not the `gpu_type` in `hosts.toml`. A host whose configured `gpu_type` disagrees is marked `*` and listed under the table, so stale labels show up instead of silently steering `-H` lists. `-t/--gpu-type` and `-m` always use the reported values.

**Example output:**

```
SERVER         GPU TYPE                 GPUs  FREE   VRAM FREE  VRAM TOTAL  USERS / PROCESSES
─────────────────────────────────────────────────────────────────────────────────────────────
cuda14         A40                         4     2     96.0 GB    192.0 GB  woodj4: python3
cuda20         L40S                        4     4    192.0 GB    192.0 GB  idle
cuda05         RTX PRO 5000 Blackwell      2     0      0.0 GB     95.6 GB  jsmith: python
...
```

---

### Cluster Operations

Run a shell command on **every configured server in parallel** (or only the `-H` selection):

```bash
task exec [-H SPEC] -- COMMAND
```

Results stream in as servers respond, with a live progress indicator. Failed servers print their full output at the end.

**Examples:**

```bash
task exec -- "cd /vol/ecrg-solar/woodj4/depth-learning && pip install -e ."   # install everywhere
task exec -- python3 --version
task exec -H cuda14,cuda15 -- nvidia-smi -L
```

---

### Hosts & Connections

| Command | Description |
|---------|-------------|
| `task sync` | Push this git repo's working tree (including uncommitted changes) to its copy on the servers. Takes `-H` and `-C`. |
| `task sync --check` | Only check whether the servers' copy matches your working tree. Exits 1 if not. |
| `task fetch PATH...` | Copy outputs back from the servers (see [Fetching Results](#fetching-results)). |
| `task hosts` | Table of configured hosts: kind, group, route (direct or via jump hosts), live reachability, GPU count, CPU slots and path mappings. Also shows the network mode and where the config came from. |
| `task connect` | Log in to each jump host interactively, once. Use this when the jump host wants a password or OTP; the connection then stays open for the daemon. |
| `task ssh HOST` | Open an interactive shell on a host through the same route the daemon uses. Useful for debugging. |
| `task config init [--user NAME]` | Write a commented `~/.gpu-spooler/hosts.toml` template for your username on the servers (default: your local one). Never overwrites an existing file. |
| `task config path` | Print which config file is in use. |

---

### Daemon

The daemon starts automatically when you submit a job. You rarely need to manage it directly.

| Command | Description |
|---------|-------------|
| `task daemon start` | Start the daemon if it isn't already running. |
| `task daemon stop` | Stop the daemon. Running tasks keep running on their servers. The next daemon picks them up again and copies the output it missed. |
| `task daemon status` | Whether it is running, and its PID. |

Daemon logs are written to `~/.gpu-spooler/daemon.log`.

---

## Configuration

Hosts are read from `~/.gpu-spooler/hosts.toml`. You can point at another file with the `GPU_SPOOLER_CONFIG` environment variable. **Without the file, the built-in ECS server list is used** with plain `ssh <name>`, exactly as before. That's all you need on an ECS machine.

Create a starting point with:

```bash
task config init          # writes ~/.gpu-spooler/hosts.toml
task hosts                # check what it resolves to
```

The template contains every ECS server. The main parts are:

```toml
[network]
mode = "auto"                  # auto | direct | jump
# probe = "cuda00.ecs.vuw.ac.nz"  # host used to detect if you're on campus

[limits]
max_gpus = 5                   # at most 5 GPUs held by your tasks at once (unset: no limit)

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
remote_root = "/vol/ecrg-solar/woodj4"   # git repos run from <remote_root>/<repo>
path_map = [ { local = "~/data", remote = "/vol/ecrg-solar/woodj4/data" } ]
hosts = ["cuda-small0", "cuda00", "cuda01", "...", "bordeaux"]

[hosts.cuda14]                 # per-host settings override the group
gpu_type = "A40"               # display only; nvidia-smi's name wins
gpu_mem_gb = 48                # -m/--min-mem fallback until the host is queried
enabled = true                 # false = never dispatch here

[hosts.local]
kind = "local"                 # run on this machine, no ssh
cpu_slots = 2
```

- **`[limits] max_gpus`** caps the GPUs your tasks hold at once, across all hosts. Queued tasks wait until running ones finish, a task asking for more than the cap is refused at submit, and `task list` shows `GPUs used/cap`. It lives in the config rather than on the command line so a forgotten flag can't exceed it.
- **Groups** give defaults to their hosts. A host's group names also act as tags, which you can use with `-H`, `task gpus -H`, and so on. You can add more with `tags = [...]`.
- **`remote_root`** is where your git repos live on those servers: see [Running Your Git Repos from a Laptop](#running-your-git-repos-from-a-laptop).
- **`path_map`** translates the working directory when a job runs on that host, using the longest matching prefix. For example, `~/code/proj` on your laptop becomes `/vol/ecrg-solar/woodj4/proj` on ECS. The spooler doesn't copy code. It only translates the path, so the code needs to exist at the remote path.
- **Jump chain** entries are either `user@host[:port]` or the name of a `[hosts.X]` entry.
- From the config, the spooler generates `~/.gpu-spooler/ssh_config.direct` and `ssh_config.jump`, and calls `ssh -F <file>`. Your own `~/.ssh/config` is included last, so settings such as keys and agents still apply.
- **`[detective]`** (optional, off by default) reads gpu-detective's history to find GPUs held but idle: processes on them, under 5% utilisation in every sweep for `idle_minutes` (default 30).
  - `task gpus` marks those GPUs `IDLE` with how long they've been idle, and adds a count to the totals. A GPU is only marked if the live query agrees.
  - `task list` flags your own running tasks whose GPUs are all idle, since they still count against `max_gpus`.
  - Placement doesn't change: a GPU with a process on it is never treated as free anyway.
  - The history is opened read-only (`~/.gpu-detective/history.db` by default, or set `history = "..."` or `GPU_DETECTIVE_DB`). gpu-spooler doesn't import gpu-detective.

  ```toml
  [detective]
  idle_minutes = 30
  ```

---

## Working Off Campus

Since Trimester 2 2025, off-campus SSH to ECS must go through `entry.ecs.vuw.ac.nz`. The route is **laptop → entry → barretts → cudaXX**. With `mode = "auto"`, the spooler checks whether it can reach the servers directly (on campus, including VUW wifi and the halls). If it can't, it goes through the jump chain. The result is cached for 5 minutes, and the daemon re-checks after that. `task hosts`, `task list` and `task gpus` show the mode in use.

Each hop keeps one shared connection (with `ControlPersist`), so the 30 servers don't each log in to `entry` separately.

- **With SSH keys** on entry (and your key on barretts and the servers), nothing else is needed.
- **With OTP + password**, run `task connect` once and type your password and code. The connection to entry stays open for `jump_control_persist` (12 h by default), and the daemon reuses it. When the daemon reports `authentication failed`, run `task connect` again.

Jobs run detached on the server. If your laptop sleeps or changes network, the jobs keep running, and the daemon catches up on their output when it can reach the servers again.

---

## Running Your Git Repos from a Laptop

On an ECS machine, your files are on the same NFS as the servers, so jobs simply run in your current directory. A laptop doesn't share that filesystem. So when you submit from inside a **git repo**, the job runs in **that repo's copy on the server** instead:

```
laptop:  ~/code/myproj/experiments      (git repo root: ~/code/myproj)
server:  /vol/ecrg-solar/woodj4/myproj/experiments
         └── remote_root ──────┘ └repo┘ └ same subdirectory ┘
```

`remote_root` is set per group (or host) in `hosts.toml`. The ECS template uses `/vol/ecrg-solar/woodj4`. The ECS servers share storage, so one copy serves the whole group.

The workflow:

```bash
cd ~/code/myproj/experiments
task sync                    # create/update the copy on the servers
task submit python3 train.py # runs in /vol/ecrg-solar/woodj4/myproj/experiments
# ... edit code ...
task submit python3 train.py # warns: "... is at abc1234, but you're at def5678 + uncommitted changes; run `task sync`"
task sync
task sync --check             # check without changing anything
```

- **Submit only checks; it never changes the copy on the server.** If the copy is missing, out of date or has edits made on the server, submit prints a warning and still queues the job. The check is one SSH call per group.
- **`task sync` pushes from your laptop**, over the same SSH route as everything else (through the jump hosts when off campus). The servers never need access to GitHub, and unpushed commits work fine.
- **Uncommitted changes are included.** The sync takes a snapshot of your working tree with `git stash create`, which doesn't touch your local repo or stash. Untracked files (data, outputs) are not copied. Commit them, or put them on the server yourself.
- On the server, the copy is a normal git repo checked out at that snapshot (detached HEAD). Files you edit on the server are never overwritten. If they conflict, `task sync` fails, and you can sort it out with `task ssh <host>`. Untracked files there, such as outputs and checkpoints, are left alone.
- Running jobs share the checkout. Syncing while they run changes the files under them, and `--sync` tells you how many are affected.
- A `path_map` entry that covers the directory takes precedence over `remote_root`. Directories outside a git repo are used unchanged, as before.

### Fetching Results

`task sync` only goes laptop → servers. Outputs come back with `task fetch`, which resolves paths the same way as jobs and `task sync`:

```bash
cd ~/code/myproj
task fetch results                     # /vol/ecrg-solar/woodj4/myproj/results → ./results
task fetch results --after 12 13       # wait for tasks 12 and 13 first; fetch only if both are done
task fetch results --dry-run           # show what would change
task fetch /vol/ecrg-solar/woodj4/run-v3/results:results    # explicit REMOTE:LOCAL
```

- A relative path is taken from the current directory and must stay inside the repo; anything else needs the explicit `REMOTE:LOCAL` form (a file fetched into an existing directory lands inside it, as with `scp`).
- **Local work is never lost.** A local file is replaced only if it is identical, unchanged since the last fetch wrote it, or tracked by git and unmodified (so git still has it). Anything else is kept and listed; `--force` replaces it. Files missing on the server are never deleted locally.
- **Links are copied as links, never followed.** A link that points outside the destination (e.g. a `data` symlink to private data elsewhere) is skipped and reported.
- Fetching while tasks still write under those paths prints a warning; use `--after` to wait for them.
- Directories are compared by SHA-256 on the server, so only changed files are transferred (one gzipped tar over the usual ssh route).

---

### Run Provenance (taura)

If [taura](https://github.com/woodRock/taura) is installed and the repo keeps a taura ledger (a `ledger/` directory), every task is also recorded as a taura run in the **local** repo's `ledger/runs/`, with no extra flags:

| When | What the daemon records |
|------|-------------------------|
| the task starts | the command, the commit the server actually checked out (a `task sync` of a dirty tree is recorded as dirty, with the commit it was made on), `--config`/`--seed` from the command, host, GPUs, start time |
| the task ends | exit code and finish time (each retry is its own run: `<date>-spooler-<id>-2`, ...) |
| `task fetch PATHS --after ID` | the files that came back new or changed, with content hashes |

taura then links the paper's results and figures to these runs (`taura trace`, `taura tui`) and warns when a result comes from a dirty or failed run, or from code that has changed since. Commands run on your own machine are recorded the same way with `taura run -- python3 train.py ...`.

For a one-off check that isn't an experiment, `task submit --no-record -- ...` skips the record.

taura is never required: without it, or for a repo without a ledger, nothing changes. To turn it off:

```toml
[provenance]
taura = "off"        # default "auto": record when taura is installed and the repo has a ledger
```

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
| cuda-small1 | 3 | RTX 4000 Ada Generation | 20 GB | 16 | 144 GB |
| cuda00 | 2 | RTX 6000 | 24 GB | 32 | 96 GB |
| cuda01–04 | 2 | RTX A5000 | 24 GB | 16 | 64 GB |
| cuda05–06 | 2 | RTX PRO 5000 Blackwell | 48 GB | 16–24 | 144 GB |
| cuda07–08 | 3 | RTX A5000 | 24 GB | 24 | 144 GB |
| cuda09 | 3 | RTX PRO 4000 Blackwell | 24 GB | 32 | 96 GB |
| cuda10–11 | 3 | RTX A5000 | 24 GB | 24 | 112–128 GB |
| cuda12–13 | 3 | RTX A5000 | 24 GB | 32 | 96 GB |
| cuda14–15 | 4 | A40 | 48 GB | 64 | 256 GB |
| cuda16 | 3 | RTX A6000 | 48 GB | 32 | 192 GB |
| cuda17 | 3 | Quadro RTX 6000 | 24 GB | 24 | 144 GB |
| cuda18 | 3 | RTX A6000 | 48 GB | 32 | 192 GB |
| cuda19 | 4 | L4 | 24 GB | 64 | 256 GB |
| cuda20 | 4 | L40S | 48 GB | 64 | 384 GB |
| cuda21–22 | 2 | RTX 6000 Ada Generation | 48 GB | 48 | 256 GB |
| gryphon | 1 | RTX A6000 | 48 GB | 64 | 192 GB |
| red-tomatoes | 2 | RTX A6000 | 48 GB | 32 | 96 GB |
| piccolo | 2 | RTX A5000 | 24 GB | 20 | 128 GB |
| the-villa | 1 | RTX A6000 | 48 GB | 64 | 192 GB |
| bordeaux | 1 | Quadro RTX 6000 | 24 GB | 20 | 64 GB |

**Total: 30 servers · 78 GPUs.** Types are the names `nvidia-smi` reported on 2026-10-17 (cuda00–04 were unreachable and are unverified); `gpu-status` shows the current ones.

---

## State & Logs

All state is stored in `~/.gpu-spooler/`:

```
~/.gpu-spooler/
├── hosts.toml          # your host configuration (optional)
├── tasks.db            # SQLite database (all task records)
├── daemon.pid          # PID of the running daemon
├── daemon.log          # Daemon stdout/stderr
├── ssh_config.direct   # generated ssh configs (don't edit)
├── ssh_config.jump
├── cm/                 # shared ssh connection sockets
├── network.json        # cached network-mode probe
├── gpu_snapshot.json   # latest GPU query from the daemon (read by tui)
├── gpu_snapshot.request  # touched by tui to ask for fresh snapshots
└── logs/
    ├── 1.log           # Output for task 1
    └── ...
```

On each server, a job lives in `~/.gpu-spooler/jobs/<id>-<random>/` (`run.sh`, `pid`, `out.log`, `rc`). ECS home directories are shared over NFS, so this is the same directory on every ECS machine. `task clear` removes these directories for finished tasks.

Task logs include a header with the command, server, GPU indices, start time, and working directory, followed by the raw stdout/stderr of your job.

---

## Troubleshooting

**Command not found on the remote server**

The daemon sources `~/.bashrc` before running your command, which should load conda environments and `~/.local/bin`. If a command is still missing, verify it's accessible in a non-interactive SSH session:

```bash
task ssh cuda14         # then: which mycommand
```

**Working directory not found (`No such file or directory`)**

NFS automount paths (e.g. `/am/...`) may not be mounted yet when the SSH session starts. The spooler runs `ls <dir>` first to trigger the automount. If this still fails, use `-d` to provide the path as seen from the remote server:

```bash
task submit -C /vol/ecrg-solar/woodj4/project -- python3 train.py
```

**All servers show as unreachable**

Run `task hosts` to see the network mode and the SSH error for each host. SSH key-based auth must be set up for the servers. If the error mentions authentication and you're off campus, run `task connect`. To test one host by hand:

```bash
ssh -F ~/.gpu-spooler/ssh_config.direct cuda01 echo ok    # or ssh_config.jump
```

If this prompts for a password, add your public key to the servers:

```bash
ssh-copy-id cuda01
```

**GPU shows as busy but no jobs are running**

`gpu-status` uses `nvidia-smi pmon` which shows all processes using the GPU, including display servers and idle CUDA contexts. Run `task gpus -d` to see the specific process names and users holding each GPU.

**Daemon disappeared while a task was running**

Start it again with `task daemon start`. Running jobs are picked up again, and the output they wrote in the meantime is copied over. Only tasks started by the old, pre-detached version of the spooler are marked `lost`.

**`task kill` couldn't reach the server**

The task is marked `killed`, but the process may still be running. Check and kill it by hand with the command printed in the warning (`task ssh <server>`, then `kill -- -<PGID>`). `task info ID` shows the PGID.

---

## Development

Tests use pytest. The integration tests start a fake cluster on localhost: four unprivileged `sshd`s (entry, barretts, cuda01 with 2 × A5000, cuda02 with 4 × A40) with a mock `nvidia-smi`. They are skipped if `sshd` isn't available.

```bash
.venv/bin/pip install -e '.[test]'      # pytest + textual (the tui tests use Textual's test pilot)
.venv/bin/python -m pytest tests

# Use the fake cluster by hand
dev/fake_cluster.sh up
export GPU_SPOOLER_CONFIG=/tmp/ts-fake-cluster/hosts.toml
export HOME=$(mktemp -d /tmp/tsh.XXXX)       # optional: isolate ~/.gpu-spooler
dev/fake_cluster.sh busy cuda01 0            # another "user" takes GPU 0
dev/fake_cluster.sh stop barretts            # simulate a dropped jump host
dev/fake_cluster.sh start barretts
dev/fake_cluster.sh down
```

The sshd logs in `/tmp/ts-fake-cluster/run/sshd_*.log` show the traffic through each hop.

**Rust port (in progress).** A Cargo workspace (`crates/gpu-spooler`, the library, and `crates/gpu-spooler-cli`, the `task` binary) is replacing the Python package one step at a time; until it's done, the Python version is the one that's installed. Its tests compare against fixtures generated from the Python code, so what is sent to the servers stays byte for byte the same:

```bash
cargo test                                  # Rust tests, including the golden fixtures
.venv/bin/python dev/gen_golden.py          # regenerate tests/golden/parity/ after changing the Python code
.venv/bin/python dev/gen_golden.py --check  # CI: are the fixtures current?
cargo build --release && GPU_SPOOLER_BIN=target/release/task \
  .venv/bin/python -m pytest tests     # the integration tests through the Rust binary, plus parity
                                       # (same output as Python) and daemon handover tests
```

Ported so far: everything but `sync`, `fetch` and `tui`, including the daemon. The two daemons adopt each other's running jobs, so switching between versions never interrupts a job. Try it with `target/release/task`: `task daemon stop` with one version, then `task daemon start` with the other.
