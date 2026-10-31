import { useEffect, useState } from "react";
import { useActions } from "./actions";
import { events } from "./bindings";
import { needsConnect } from "./Banners";
import { Shortcuts, useKeys } from "./keys";
import { GpuView } from "./GpuView";
import { HostsView } from "./HostsView";
import { NewTask } from "./NewTask";
import { QueueView } from "./QueueView";
import { UiProvider } from "./ui";
import { useGpus, useQueue } from "./useQueue";

type View = "queue" | "gpus" | "hosts";

export function App() {
  return (
    <UiProvider>
      <Main />
    </UiProvider>
  );
}

function Main() {
  const queue = useQueue();
  const gpus = useGpus();
  const [view, setView] = useState<View>("queue");
  const [selected, setSelected] = useState<number | null>(null);
  const [newTask, setNewTask] = useState(false);
  const [help, setHelp] = useState(false);
  const actions = useActions();
  useKeys(
    (e) => {
      if (e.key === "?") setHelp(true);
    },
    [],
  );
  // ⌘1–3 switch views and ⌘N is a new task, from anywhere but a dialog.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!e.metaKey || e.shiftKey || e.altKey || document.querySelector(".overlay")) return;
      const views: Record<string, View> = { "1": "queue", "2": "gpus", "3": "hosts" };
      if (views[e.key]) {
        e.preventDefault();
        setView(views[e.key]);
      } else if (e.key.toLowerCase() === "n") {
        e.preventDefault();
        setNewTask(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  // From the menu bar: show a task, or the New Task sheet.
  useEffect(() => {
    const unlisten = events.showRequest.listen(({ payload }) => {
      if (payload.task != null) {
        setView("queue");
        setSelected(payload.task);
      }
      if (payload.new_task) setNewTask(true);
    });
    return () => {
      unlisten.then((f) => f());
    };
  }, []);
  if (!queue || !gpus) return <div className="loading">Reading the queue…</div>;

  const daemonPid = queue.daemon_pid;
  const running = queue.tasks.filter((t) => t.status === "running").length;
  const free = gpus.hosts.flatMap((h) => (h.reachable ? h.gpus : [])).filter((g) => g.state === "free").length;
  const nav: [View, string, string | number | null, boolean][] = [
    ["queue", "Queue", running || null, false],
    ["gpus", "GPUs", gpus.ts != null ? free : null, needsConnect(gpus)],
    ["hosts", "Hosts", null, needsConnect(gpus)],
  ];
  const showTask = (id: number) => {
    setNewTask(false);
    setView("queue");
    setSelected(id);
  };

  return (
    <div className="shell">
      <nav className="sidebar">
        <div className="brand">gpu-spooler</div>
        {nav.map(([id, label, badge, alert]) => (
          <button
            key={id}
            className={`nav ${view === id ? "on" : ""}`}
            onClick={() => setView(id)}
            title={`${label} (⌘${nav.findIndex((n) => n[0] === id) + 1})`}
          >
            <span>{label}</span>
            {alert ? <span className="dot err" /> : badge != null && <span className="badge">{badge}</span>}
          </button>
        ))}
        <button className="btn primary new-task" onClick={() => setNewTask(true)} title="New task (⌘N)">
          New task
        </button>
        <div className="sidebar-foot">
          <button className="link keys-hint" onClick={() => setHelp(true)}>
            Keyboard shortcuts <kbd>?</kbd>
          </button>
          {daemonPid != null ? (
            <span title={`PID ${daemonPid}`}>
              <span className="dot ok" /> Daemon running
            </span>
          ) : (
            <>
              <span>
                <span className="dot warn" /> Daemon stopped
              </span>
              <button className="link" disabled={actions.busy} onClick={actions.startDaemon}>
                Start
              </button>
            </>
          )}
        </div>
      </nav>
      {view === "queue" && (
        <QueueView state={queue} selected={selected} setSelected={setSelected} onNewTask={() => setNewTask(true)} />
      )}
      {view === "gpus" && <GpuView gpus={gpus} daemonPid={daemonPid} />}
      {view === "hosts" && <HostsView gpus={gpus} daemonPid={daemonPid} />}
      {help && <Shortcuts onClose={() => setHelp(false)} />}
      {newTask && <NewTask queue={queue} gpus={gpus} onClose={() => setNewTask(false)} onQueued={showTask} />}
    </div>
  );
}
