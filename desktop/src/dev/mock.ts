// A browser preview for design work: `npm run dev`, then open http://localhost:1420 (add
// ?window=popover for the popover). The backend is replaced by made-up data, so every screen
// renders without Tauri. Only loaded in dev builds outside Tauri; never in the app.
import { mockIPC, mockWindows } from "@tauri-apps/api/mocks";
import type { GpuState, QueueState, Task } from "../types";

const now = Date.now() / 1000;

function task(id: number, status: string, fields: Partial<Task> = {}): Task {
  return {
    id,
    command: "python3 train.py --epochs 50",
    num_gpus: 1,
    min_mem: 0,
    status,
    server: null,
    gpu_indices: null,
    ssh_pid: null,
    submitted_at: now - 3600 + id * 60,
    started_at: null,
    finished_at: null,
    exit_code: null,
    log_file: null,
    working_dir: "/Users/you/code/project",
    name: null,
    retry_count: 0,
    job_key: null,
    remote_pid: null,
    log_offset: 0,
    host_spec: null,
    remote_dir: null,
    last_seen: null,
    repo_root: null,
    gpu_type: null,
    git_commit: null,
    git_dirty: null,
    git_dirty_paths: null,
    no_record: 0,
    ...fields,
  };
}

const ran = (start: number, end: number | null, server: string, gpus = "0") => ({
  started_at: now - start,
  finished_at: end === null ? null : now - end,
  server,
  gpu_indices: gpus,
});

const tasks: Task[] = [
  task(14, "queued", { name: "sweep lr=3e-4", host_spec: "cuda14,cuda15", gpu_type: "A40" }),
  task(13, "running", { name: "sweep lr=1e-3", ...ran(780, null, "cuda15", "2"), command: "python3 sweep.py --lr 1e-3" }),
  task(12, "running", { name: "pretrain", num_gpus: 2, ...ran(5400, null, "cuda14", "0,1"), command: "torchrun --nproc 2 pretrain.py" }),
  task(11, "failed", { name: "eval", exit_code: 1, ...ran(4000, 3900, "cuda08"), command: "python3 eval.py --split test" }),
  task(10, "done", { name: "baseline", ...ran(9000, 6100, "cuda12"), exit_code: 0 }),
  task(9, "killed", { ...ran(9800, 9500, "cuda05"), command: "python3 train.py --epochs 500" }),
  task(8, "cancelled", { name: "duplicate", host_spec: "local" }),
  task(7, "lost", { name: "overnight", ...ran(60000, 30000, "cuda02") }),
  task(6, "done", { name: "plots", num_gpus: 0, ...ran(70000, 69900, "local", ""), command: "python3 make_plots.py && echo done" }),
];

const queue: QueueState = { tasks, daemon_pid: 4242, gpus_in_use: 3, max_gpus: 5, error: null };

const gpu = (index: number, state: "free" | "ours" | "used", extra: Record<string, unknown> = {}) => ({
  index,
  name: "NVIDIA A40",
  mem_used_mb: state === "free" ? 2 : 31000,
  mem_total_mb: 46068,
  utilization_pct: state === "free" ? 0 : 87,
  state,
  task_id: null as number | null,
  processes: state === "used" ? [{ pid: 1, user: "alex", mem_used_mb: 31000, command: "python3" }] : [],
  idle_seconds: null as number | null,
  ...extra,
});

const host = (name: string, gpus: ReturnType<typeof gpu>[], reachable: boolean | null = true, extra = {}) => ({
  name,
  group: "ecs",
  role: "worker",
  route: "ecs",
  enabled: true,
  configured: true,
  cpu_slots: 0,
  remote_root: "/vol/ecrg-solar/you",
  gpu_type: "A40",
  reachable,
  error: reachable === false ? "ssh exited 255" : "",
  gpus,
  ...extra,
});

const gpus: GpuState = {
  ts: now - 12,
  mode: "jump",
  hosts: [
    host("cuda14", [gpu(0, "ours", { task_id: 12 }), gpu(1, "ours", { task_id: 12 }), gpu(2, "free"), gpu(3, "used")]),
    host("cuda15", [gpu(0, "used"), gpu(1, "used", { idle_seconds: 5400, utilization_pct: 0 }), gpu(2, "ours", { task_id: 13 }), gpu(3, "free")]),
    host("cuda08", [gpu(0, "free", { name: "NVIDIA RTX A5000", mem_total_mb: 24564 }), gpu(1, "used", { name: "NVIDIA RTX A5000", mem_total_mb: 24564 })]),
    host("cuda12", [gpu(0, "free", { name: "NVIDIA RTX A5000", mem_total_mb: 24564 }), gpu(1, "free", { name: "NVIDIA RTX A5000", mem_total_mb: 24564 }), gpu(2, "used", { name: "NVIDIA RTX A5000", mem_total_mb: 24564 })]),
    host("cuda02", [], false),
    host("cuda05", [], false),
    host("local", [], null, { group: null, route: "local", cpu_slots: 2, remote_root: null, gpu_type: null }),
    host("entry", [], null, { role: "jump", group: null, route: "direct", remote_root: null, gpu_type: null }),
  ],
} as GpuState;

const log = [
  "# task 13: python3 sweep.py --lr 1e-3",
  "# server: cuda15  CUDA_VISIBLE_DEVICES: 2",
  "──────────────────────────────────────────",
  "\x1b[1mEpoch 1/50\x1b[0m",
  "train loss 2.3011  acc 0.112",
  "\x1b[33mwarning:\x1b[0m gradient norm 41.2 clipped to 5.0",
  "\x1b[1mEpoch 2/50\x1b[0m",
  "train loss 1.8807  acc 0.341",
  "  12%|█▏        | 60/500 [00:31<03:48]\r  48%|████▊     | 240/500 [02:04<02:15]\r 100%|██████████| 500/500 [04:19<00:00]",
  "\x1b[32mval acc 0.402\x1b[0m (best)",
  "\x1b[31mERROR\x1b[0m checkpoint dir not writable, retrying in 5 s",
  "saved checkpoints/lr1e-3/epoch2.pt",
  "",
].join("\n");

export function install() {
  const label = new URLSearchParams(location.search).get("window") ?? "main";
  mockWindows(label, ...["main", "popover"].filter((l) => l !== label));
  mockIPC(
    (cmd, args) => {
      const a = args as Record<string, unknown>;
      switch (cmd) {
        case "queue_state":
          return queue;
        case "gpu_state":
          return gpus;
        case "task_details": {
          const t = tasks.find((x) => x.id === a.id)!;
          return [
            ["Status", t.status],
            ["Label", t.name ?? ""],
            ["Command", t.command],
            ["GPUs req", String(t.num_gpus)],
            ["Working dir", t.working_dir],
            ["Server", t.server ?? "-"],
          ];
        }
        case "log_tail":
          return { text: log, offset: log.length, truncated: false, reset: true, missing: false, path: `~/.gpu-spooler/logs/${a.id}.log` };
        case "log_since":
          return { text: "", offset: a.offset, truncated: false, reset: false, missing: false, path: "" };
        case "launch_at_login":
          return false;
        case "plugin:window|is_visible":
          return true;
        default:
          return null;
      }
    },
    { shouldMockEvents: true },
  );
}
