import { GpuBanners } from "./Banners";
import { model } from "./GpuView";
import type { GpuState, HostView } from "./types";
import { useGpuKeepAlive, useNow } from "./useQueue";

export function HostsView({ gpus, daemonPid }: { gpus: GpuState; daemonPid: number | null }) {
  useGpuKeepAlive(true);
  const now = useNow(true, 5000);
  const workers = gpus.hosts.filter((h) => h.role === "worker");
  const jumps = gpus.hosts.filter((h) => h.role === "jump");
  return (
    <div className="view">
      <header className="bar">
        <h1>Hosts</h1>
        <div className="chips">
          <span className="chip">
            <strong>{workers.filter((h) => h.enabled).length}</strong> enabled
            {jumps.length > 0 && (
              <>
                {" "}
                · <strong>{jumps.length}</strong> jump
              </>
            )}
          </span>
        </div>
      </header>
      <GpuBanners gpus={gpus} daemonPid={daemonPid} now={now} />
      <main className="table-wrap">
        {gpus.hosts.length === 0 ? (
          <div className="empty">
            No hosts configured. Create <code>~/.gpu-spooler/hosts.toml</code> with <code>task config init</code>.
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Host</th>
                <th>Status</th>
                <th>Group</th>
                <th>Route</th>
                <th>GPUs</th>
                <th className="num">CPU slots</th>
                <th>Remote root</th>
              </tr>
            </thead>
            <tbody>
              {[...workers, ...jumps].map((h) => (
                <Row key={h.name} host={h} />
              ))}
            </tbody>
          </table>
        )}
      </main>
    </div>
  );
}

function Row({ host: h }: { host: HostView }) {
  const free = h.gpus.filter((g) => g.state === "free").length;
  const models = [...new Set(h.gpus.map((g) => model(g.name)))].join(", ");
  let status;
  if (h.role === "jump") status = <span className="muted">jump host</span>;
  else if (!h.enabled) status = <span className="muted">disabled</span>;
  else if (h.reachable === null) status = <span className="muted">not polled</span>;
  else if (h.reachable) status = <span className="status s-running">up</span>;
  else
    status = (
      <span className="status s-failed" title={h.error}>
        unreachable
      </span>
    );
  return (
    <tr className={!h.enabled ? "dim" : ""}>
      <td className="nowrap">
        <strong>{h.name}</strong>
        {!h.configured && <span className="muted small"> (not in hosts.toml)</span>}
      </td>
      <td>
        {status}
        {h.reachable === false && h.error && <div className="muted small">{h.error}</div>}
      </td>
      <td className="muted">{h.group ?? "–"}</td>
      <td className="muted nowrap">{h.route === "direct" || h.route === "local" ? h.route : `via ${h.route}`}</td>
      <td>
        {h.gpus.length > 0 ? (
          <>
            {free}/{h.gpus.length} free <span className="muted small">{models}</span>
          </>
        ) : h.gpu_type ? (
          <span className="muted">{h.gpu_type} (configured)</span>
        ) : (
          <span className="muted">–</span>
        )}
      </td>
      <td className="num">{h.cpu_slots || <span className="muted">–</span>}</td>
      <td className="mono muted small nowrap">{h.remote_root ?? "–"}</td>
    </tr>
  );
}
