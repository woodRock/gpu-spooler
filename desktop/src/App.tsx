import { useState } from "react";
import { needsConnect } from "./Banners";
import { GpuView } from "./GpuView";
import { HostsView } from "./HostsView";
import { QueueView } from "./QueueView";
import { useGpus, useQueue } from "./useQueue";

type View = "queue" | "gpus" | "hosts";

export function App() {
  const queue = useQueue();
  const gpus = useGpus();
  const [view, setView] = useState<View>("queue");
  if (!queue || !gpus) return <div className="loading">Reading the queue…</div>;

  const daemonPid = queue.daemon_pid;
  const running = queue.tasks.filter((t) => t.status === "running").length;
  const free = gpus.hosts.flatMap((h) => (h.reachable ? h.gpus : [])).filter((g) => g.state === "free").length;
  const nav: [View, string, string | number | null, boolean][] = [
    ["queue", "Queue", running || null, false],
    ["gpus", "GPUs", gpus.ts != null ? free : null, needsConnect(gpus)],
    ["hosts", "Hosts", null, needsConnect(gpus)],
  ];

  return (
    <div className="shell">
      <nav className="sidebar">
        <div className="brand">gpu-spooler</div>
        {nav.map(([id, label, badge, alert]) => (
          <button key={id} className={`nav ${view === id ? "on" : ""}`} onClick={() => setView(id)}>
            <span>{label}</span>
            {alert ? <span className="dot err" /> : badge != null && <span className="badge">{badge}</span>}
          </button>
        ))}
        <div className="sidebar-foot">
          {daemonPid != null ? (
            <span title={`PID ${daemonPid}`}>
              <span className="dot ok" /> Daemon running
            </span>
          ) : (
            <span>
              <span className="dot warn" /> Daemon stopped
            </span>
          )}
        </div>
      </nav>
      {view === "queue" && <QueueView state={queue} />}
      {view === "gpus" && <GpuView gpus={gpus} daemonPid={daemonPid} />}
      {view === "hosts" && <HostsView gpus={gpus} daemonPid={daemonPid} />}
    </div>
  );
}
