import { useEffect, useMemo, useRef, useState } from "react";
import { useActions } from "./actions";
import { DaemonBanner } from "./Banners";
import { ACTIVE, fmtDuration, fmtTs, runtime } from "./format";
import { useKeys } from "./keys";
import { TaskPane } from "./TaskPane";
import type { QueueState, Task } from "./types";
import { useNow } from "./useQueue";

type Filter = "all" | "active" | "finished";

export function QueueView({
  state,
  selected,
  setSelected,
  onNewTask,
}: {
  state: QueueState;
  selected: number | null;
  setSelected: (id: number | null) => void;
  onNewTask: () => void;
}) {
  const [filter, setFilter] = useState<Filter>("all");
  const [search, setSearch] = useState("");
  const actions = useActions();
  const finished = state.tasks.filter((t) => !ACTIVE.includes(t.status)).length;
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
  const task = state.tasks.find((t) => t.id === selected) ?? null;

  const searchBox = useRef<HTMLInputElement>(null);

  // As in `task tui`: ↑/↓ move the selection, Esc closes the pane, k kills, r resubmits, n is a new
  // task; / filters. Not while typing, or while a dialog is open (see keys.tsx).
  useKeys(
    (e) => {
      if (e.key === "Escape") setSelected(null);
      if (e.key === "n") onNewTask();
      if (e.key === "/") {
        e.preventDefault();
        searchBox.current?.focus();
      }
      if (task && e.key === "k" && ACTIVE.includes(task.status)) actions.kill(task);
      if (task && e.key === "r") actions.resubmit(task, setSelected);
      if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
      e.preventDefault();
      const i = shown.findIndex((t) => t.id === selected);
      const next = e.key === "ArrowDown" ? Math.min(i + 1, shown.length - 1) : Math.max(i - 1, 0);
      if (shown[next]) setSelected(shown[next].id);
    },
    [shown, selected, task, actions, onNewTask, setSelected],
  );
  // ⌘F filters too, as in any Mac app.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey && e.key.toLowerCase() === "f" && !document.querySelector(".overlay")) {
        e.preventDefault();
        searchBox.current?.focus();
        searchBox.current?.select();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => {
    if (selected !== null) document.getElementById(`task-${selected}`)?.scrollIntoView({ block: "nearest" });
  }, [selected]);

  return (
    <div className="view">
      <header className="bar">
        <h1>Queue</h1>
        <div className="chips">
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
      <DaemonBanner pid={state.daemon_pid} what="queued tasks will not start" />

      <div className="toolbar">
        <div className="segmented" role="tablist">
          {(["all", "active", "finished"] as Filter[]).map((f) => (
            <button key={f} role="tab" aria-selected={filter === f} onClick={() => setFilter(f)}>
              {f[0].toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
        <input
          ref={searchBox}
          className="search"
          type="search"
          placeholder="Filter by label, command, server…  /"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          onKeyDown={(e) => {
            // Esc clears the filter, then leaves the box (so ↑/↓ move through the rows again).
            if (e.key !== "Escape") return;
            if (search) setSearch("");
            else e.currentTarget.blur();
          }}
        />
        <button className="btn" disabled={finished === 0 || actions.busy} onClick={() => actions.clear(finished)}>
          Clear finished
        </button>
        <button className="btn primary" onClick={onNewTask} title="New task (n)">
          New task
        </button>
      </div>

      <div className={`split ${task ? "open" : ""}`}>
        <main className="table-wrap">
          {shown.length === 0 ? (
            <div className="empty">
              {state.tasks.length === 0 ? (
                <>
                  No tasks yet. Queue one with <code>task submit -- python3 train.py</code>.
                </>
              ) : filter === "active" && !search ? (
                "Nothing queued or running."
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
                  <Row
                    key={t.id}
                    task={t}
                    now={now}
                    selected={t.id === selected}
                    onSelect={() => setSelected(t.id === selected ? null : t.id)}
                  />
                ))}
              </tbody>
            </table>
          )}
        </main>
        {task && <TaskPane task={task} onClose={() => setSelected(null)} onQueued={setSelected} />}
      </div>
    </div>
  );
}

function Row({ task: t, now, selected, onSelect }: { task: Task; now: number; selected: boolean; onSelect: () => void }) {
  const gpus = t.num_gpus === 0 ? "CPU" : t.gpu_indices ? `GPU ${t.gpu_indices}` : `${t.num_gpus} GPU`;
  const exit = t.status === "failed" && t.exit_code != null ? ` (exit ${t.exit_code})` : "";
  return (
    <tr id={`task-${t.id}`} className={selected ? "selected" : ""} onClick={onSelect}>
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
      <td className="muted nowrap">{fmtTs(t.submitted_at)}</td>
    </tr>
  );
}
