import { GpuBanners } from "./Banners";
import { fmtDuration } from "./format";
import type { GpuCell, GpuState, HostView } from "./types";
import { useGpuKeepAlive, useNow } from "./useQueue";

/** As `nvidia-smi` names it, without the vendor ("NVIDIA A40" → "A40"). */
export function model(name: string): string {
  return name.replace(/^NVIDIA\s+/i, "");
}

export function GpuView({ gpus, daemonPid }: { gpus: GpuState; daemonPid: number | null }) {
  useGpuKeepAlive(true);
  const now = useNow(true, 5000);
  const polled = gpus.hosts.filter((h) => h.reachable !== null && h.role === "worker");
  const up = polled.filter((h) => h.reachable && h.gpus.length > 0);
  const cpuOnly = polled.filter((h) => h.reachable && h.gpus.length === 0);
  const down = polled.filter((h) => !h.reachable);
  const cells = up.flatMap((h) => h.gpus);
  const count = (s: string) => cells.filter((g) => g.state === s).length;

  return (
    <div className="view">
      <header className="bar">
        <h1>GPUs</h1>
        <div className="chips">
          <span className="chip">
            <strong>{count("free")}</strong> free · <strong>{count("ours")}</strong> yours · <strong>{count("used")}</strong>{" "}
            in use
          </span>
          <span className="chip">
            <strong>{up.length + cpuOnly.length}</strong> of {polled.length} servers up
          </span>
          {gpus.ts != null && (
            <span className="chip" title="When the daemon last queried the servers, and how it reached them">
              {fmtDuration(Math.max(0, now - gpus.ts))} ago · {gpus.mode ?? "?"}
            </span>
          )}
        </div>
      </header>
      <GpuBanners gpus={gpus} daemonPid={daemonPid} now={now} />

      <div className="legend">
        <span className="swatch free" /> free
        <span className="swatch ours" /> yours
        <span className="swatch used" /> someone else's
        <span className="swatch idle" /> held but idle
      </div>

      <main className="scroll">
        {polled.length === 0 && <div className="empty">No GPU data yet.</div>}
        <div className="hosts-grid">
          {up.map((h) => (
            <HostRow key={h.name} host={h} />
          ))}
        </div>
        {cpuOnly.length > 0 && (
          <p className="muted small footnote">CPU only: {cpuOnly.map((h) => h.name).join(", ")}</p>
        )}
        {down.length > 0 && (
          <div className="down">
            <h2>Unreachable ({down.length})</h2>
            <div className="down-list">
              {down.map((h) => (
                <span key={h.name} className="down-host" title={h.error || "unreachable"}>
                  {h.name}
                </span>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function HostRow({ host: h }: { host: HostView }) {
  const free = h.gpus.filter((g) => g.state === "free").length;
  const models = [...new Set(h.gpus.map((g) => model(g.name)))].join(", ");
  const gb = Math.round((h.gpus[0]?.mem_total_mb ?? 0) / 1024);
  return (
    <div className="host-row">
      <div className="host-name">
        <strong>{h.name}</strong>
        <div className="muted small">
          {models} · {gb} GB
        </div>
        <div className="small">
          <strong>{free}</strong>
          <span className="muted">/{h.gpus.length} free</span>
        </div>
      </div>
      <div className="cells">
        {h.gpus.map((g) => (
          <Cell key={g.index} gpu={g} />
        ))}
      </div>
    </div>
  );
}

function Cell({ gpu: g }: { gpu: GpuCell }) {
  const users = [...new Set(g.processes.map((p) => p.user).filter((u) => u && u !== "?"))];
  const label = g.state === "ours" ? `#${g.task_id}` : g.state === "used" ? users[0] ?? "in use" : "free";
  const mem = g.mem_total_mb > 0 ? g.mem_used_mb / g.mem_total_mb : 0;
  const idle = g.idle_seconds != null;
  const tip = [
    `GPU ${g.index} · ${model(g.name)}`,
    `${(g.mem_used_mb / 1024).toFixed(1)} / ${(g.mem_total_mb / 1024).toFixed(1)} GB`,
    g.utilization_pct >= 0 ? `${g.utilization_pct}% busy` : null,
    idle ? `idle for ${fmtDuration(g.idle_seconds)}` : null,
    ...g.processes.map((p) => `${p.user}: ${p.command} (${(p.mem_used_mb / 1024).toFixed(1)} GB)`),
  ]
    .filter(Boolean)
    .join("\n");
  return (
    <div className={`cell ${g.state} ${idle ? "idle" : ""}`} title={tip}>
      <div className="cell-top">
        <span className="cell-index">{g.index}</span>
        {g.utilization_pct >= 0 && g.state !== "free" && <span className="cell-util">{g.utilization_pct}%</span>}
      </div>
      <div className="cell-label">{label}</div>
      <div className="cell-mem">
        <div style={{ width: `${Math.round(mem * 100)}%` }} />
      </div>
    </div>
  );
}
