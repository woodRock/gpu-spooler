import { useMemo, useState } from "react";
import type { QueueState_Serialize as QueueState, Task_Serialize as Task } from "./bindings";
import { ACTIVE, fmtDuration, fmtTs, runtime } from "./format";
import { useNow, useQueue } from "./useQueue";

type Filter = "all" | "active" | "finished";

export function App() {
  const state = useQueue();
  if (!state) return <div className="loading">Reading the queue…</div>;
  return <Queue state={state} />;
}

function Queue({ state }: { state: QueueState }) {
  const [filter, setFilter] = useState<Filter>("all");
  const [search, setSearch] = useState("");
  const running = state.tasks.filter((t) => t.status === "running").length;
  const queued = state.tasks.filter((t) => t.status === "queued").length;
  const now = useNow(running > 0);

  const shown = useMemo(() => {
    const words = search.toLowerCase().split(/\s+/).filter(Boolean);
    return state.tasks.filter((t) => {
      if (filter === "active" && !ACTIVE.includes(t.status)) return false;
      if (filter === "finished" && ACTIVE.includes(t.status)) return false;
      const text = [t.id, t.name, t.command, t.server, t.host_spec, t.status].join(" ").toLowerCase();
      return words.every((w) => text.includes(w));
    });
  }, [state.tasks, filter, search]);

  return (
    <div className="app">
      <header className="bar">
        <h1>Queue</h1>
        <div className="chips">
          <Daemon pid={state.daemon_pid ?? null} />
          <span className="chip">
            <strong>{running}</strong> running · <strong>{queued}</strong> queued
          </span>
          <span className="chip" title="GPUs held by your running tasks, of [limits] max_gpus">
            GPUs <strong>{state.gpus_in_use}</strong>
            {state.max_gpus != null && <> / {state.max_gpus}</>}
          </span>
        </div>
      </header>

      {state.error && <div className="banner error">{state.error}</div>}
      {state.daemon_pid == null && (
        <div className="banner warn">
          The daemon is not running, so queued tasks will not start. Start it with <code>task daemon start</code>.
        </div>
      )}

      <div className="toolbar">
        <div className="segmented" role="tablist">
          {(["all", "active", "finished"] as Filter[]).map((f) => (
            <button key={f} role="tab" aria-selected={filter === f} onClick={() => setFilter(f)}>
              {f[0].toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
        <input
          className="search"
          type="search"
          placeholder="Filter by label, command, server…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <main className="table-wrap">
        {shown.length === 0 ? (
          <div className="empty">
            {state.tasks.length === 0 ? (
              <>
                No tasks yet. Queue one with <code>task submit -- python3 train.py</code>.
              </>
            ) : (
              "No tasks match."
            )}
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th className="num">ID</th>
                <th>Status</th>
                <th>Task</th>
                <th>Hosts</th>
                <th>Server</th>
                <th className="num">Runtime</th>
                <th>Submitted</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((t) => (
                <Row key={t.id} task={t} now={now} />
              ))}
            </tbody>
          </table>
        )}
      </main>
    </div>
  );
}

function Daemon({ pid }: { pid: number | null }) {
  return pid != null ? (
    <span className="chip" title={`PID ${pid}`}>
      <span className="dot ok" /> Daemon running
    </span>
  ) : (
    <span className="chip">
      <span className="dot warn" /> Daemon stopped
    </span>
  );
}

function Row({ task: t, now }: { task: Task; now: number }) {
  const gpus = t.num_gpus === 0 ? "CPU" : t.gpu_indices ? `GPU ${t.gpu_indices}` : `${t.num_gpus} GPU`;
  const exit = t.status === "failed" && t.exit_code != null ? ` (exit ${t.exit_code})` : "";
  return (
    <tr>
      <td className="num id">{t.id}</td>
      <td>
        <span className={`status s-${t.status}`}>
          {t.status}
          {exit}
        </span>
      </td>
      <td className="task">
        {t.name && <div className="label">{t.name}</div>}
        <div className="command" title={t.command}>
          {t.command}
        </div>
      </td>
      <td className="muted">{t.host_spec || "any"}</td>
      <td>
        {t.server ?? <span className="muted">–</span>}
        <div className="muted small">{gpus}</div>
      </td>
      <td className="num">{fmtDuration(runtime(t, now))}</td>
      <td className="muted">{fmtTs(t.submitted_at)}</td>
    </tr>
  );
}
