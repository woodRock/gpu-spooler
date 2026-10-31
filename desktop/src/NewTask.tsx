import { open } from "@tauri-apps/plugin-dialog";
import { useMemo, useState } from "react";
import { commands } from "./bindings";
import { model } from "./GpuView";
import type { GpuState, QueueState } from "./types";
import { Modal, useUi } from "./ui";

const LAST_DIR = "newTask.dir";

function lastDir(): string {
  try {
    return localStorage.getItem(LAST_DIR) ?? "";
  } catch {
    return "";
  }
}

/** The new-task form: the same fields as `task submit -g/-t/-m/-H/-n/-C`. Queueing takes two
 * steps, the form and then a review of what will run. */
export function NewTask({
  queue,
  gpus,
  onClose,
  onQueued,
}: {
  queue: QueueState;
  gpus: GpuState;
  onClose: () => void;
  onQueued: (id: number) => void;
}) {
  const { toast } = useUi();
  const [command, setCommand] = useState("");
  const [numGpus, setNumGpus] = useState(1);
  const [gpuType, setGpuType] = useState("");
  const [minMem, setMinMem] = useState(0);
  const [hosts, setHosts] = useState<string[]>([]);
  const [name, setName] = useState("");
  const [dir, setDir] = useState(lastDir);
  const [review, setReview] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // GPU models as the servers report them, with how many are free now.
  const models = useMemo(() => {
    const m = new Map<string, number>();
    for (const h of gpus.hosts)
      for (const g of h.reachable ? h.gpus : [])
        m.set(model(g.name), (m.get(model(g.name)) ?? 0) + (g.state === "free" ? 1 : 0));
    return [...m.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [gpus]);
  const workers = gpus.hosts.filter((h) => h.role === "worker" && h.enabled);
  const groups = [...new Set(workers.map((h) => h.group).filter((g): g is string => !!g))];

  const cap = queue.max_gpus;
  const overCap = cap != null && numGpus > cap;
  const waits = cap != null && !overCap && queue.gpus_in_use + numGpus > cap;
  const missing = !command.trim() ? "a command" : !dir.trim() ? "a directory" : null;

  const toggleHost = (h: string) => setHosts((cur) => (cur.includes(h) ? cur.filter((x) => x !== h) : [...cur, h]));
  const choose = async () => {
    const picked = await open({ directory: true, defaultPath: dir || undefined, title: "Working directory" });
    if (typeof picked === "string") setDir(picked);
  };

  const submit = async () => {
    setBusy(true);
    setError(null);
    const r = await commands.submitTask({
      command: command.trim(),
      num_gpus: numGpus,
      min_mem: minMem,
      gpu_type: gpuType || null,
      host_spec: hosts.length ? hosts.join(",") : null,
      name: name.trim() || null,
      working_dir: dir.trim(),
    });
    setBusy(false);
    if (r.status === "error") {
      setError(r.error);
      setReview(false);
      return;
    }
    try {
      localStorage.setItem(LAST_DIR, dir.trim());
    } catch {
      // only a convenience
    }
    toast("ok", `Queued task #${r.data.task_id}.`);
    r.data.warnings.forEach((w) => toast("warn", w));
    onQueued(r.data.task_id);
  };

  const resources = [
    numGpus === 0 ? "as a CPU job" : `on ${numGpus} GPU${numGpus === 1 ? "" : "s"}`,
    gpuType && `(${gpuType})`,
    minMem > 0 && `with at least ${minMem} GB each`,
    hosts.length ? `on ${hosts.join(", ")}` : "on any host",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <Modal onClose={() => !busy && onClose()} className="sheet">
      <h2>{review ? "Queue this task?" : "New task"}</h2>
      {review ? (
        <div className="dialog-body">
          <code className="block">{command.trim()}</code>
          <p>
            Runs {resources}, in <span className="mono">{dir.trim()}</span>
            {name.trim() && (
              <>
                , labelled <strong>{name.trim()}</strong>
              </>
            )}
            .
          </p>
          {waits && (
            <p className="note warn">
              It waits until your running tasks free GPUs: {queue.gpus_in_use} of your {cap} are in use.
            </p>
          )}
          {queue.daemon_pid == null && <p className="note">The daemon is not running; queueing starts it.</p>}
        </div>
      ) : (
        <div className="form">
          <label className="field wide">
            <span>Command</span>
            <textarea
              autoFocus
              rows={2}
              className="mono"
              placeholder="python3 train.py --epochs 50"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
            />
            <small>One line of shell, run in a login shell on the server (as task submit 'a && b').</small>
          </label>

          <label className="field">
            <span>GPUs</span>
            <input type="number" min={0} max={16} value={numGpus} onChange={(e) => setNumGpus(Math.max(0, Number(e.target.value) || 0))} />
            <small>{numGpus === 0 ? "A CPU job: runs on hosts with CPU slots." : overCap ? `More than your cap of ${cap}.` : " "}</small>
          </label>
          <label className="field">
            <span>GPU model</span>
            <select value={gpuType} onChange={(e) => setGpuType(e.target.value)} disabled={numGpus === 0}>
              <option value="">Any</option>
              {models.map(([m, free]) => (
                <option key={m} value={m}>
                  {m} ({free} free)
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Min memory (GB)</span>
            <input type="number" min={0} value={minMem} onChange={(e) => setMinMem(Math.max(0, Number(e.target.value) || 0))} disabled={numGpus === 0} />
          </label>

          <div className="field wide">
            <span>Hosts {hosts.length === 0 && <small className="inline">any</small>}</span>
            <div className="chips-select">
              {groups.map((g) => (
                <button key={g} type="button" className={`pick group ${hosts.includes(g) ? "on" : ""}`} onClick={() => toggleHost(g)}>
                  {g}
                </button>
              ))}
              {workers.map((h) => (
                <button key={h.name} type="button" className={`pick ${hosts.includes(h.name) ? "on" : ""}`} onClick={() => toggleHost(h.name)}>
                  {h.name}
                </button>
              ))}
            </div>
          </div>

          <label className="field">
            <span>Label</span>
            <input type="text" placeholder="optional" value={name} onChange={(e) => setName(e.target.value)} />
          </label>
          <label className="field wider">
            <span>Directory</span>
            <div className="with-button">
              <input type="text" className="mono" placeholder="/Users/you/project" value={dir} onChange={(e) => setDir(e.target.value)} />
              <button type="button" className="btn" onClick={choose}>
                Choose…
              </button>
            </div>
            <small>A git repo runs from its copy on the servers (task sync); other paths through path_map.</small>
          </label>
          {error && <div className="banner error wide">{error}</div>}
        </div>
      )}
      <div className="dialog-buttons">
        {review ? (
          <>
            <button className="btn" onClick={() => setReview(false)} disabled={busy}>
              Back
            </button>
            <button className="btn primary" onClick={submit} disabled={busy} autoFocus>
              {busy ? "Queueing…" : "Queue task"}
            </button>
          </>
        ) : (
          <>
            <button className="btn" onClick={onClose}>
              Cancel
            </button>
            <button className="btn primary" disabled={!!missing || overCap} title={missing ? `Needs ${missing}` : undefined} onClick={() => setReview(true)}>
              Review…
            </button>
          </>
        )}
      </div>
    </Modal>
  );
}
