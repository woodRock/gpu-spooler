import { useState } from "react";
import { useActions } from "./actions";
import { commands } from "./bindings";
import { fmtDuration } from "./format";
import type { GpuState } from "./types";

/** Older than this, the GPU view says so: the daemon refreshes every 30 s while asked. */
export const STALE_SECONDS = 120;

export function DaemonBanner({ pid, what }: { pid: number | null; what: string }) {
  const actions = useActions();
  if (pid != null) return null;
  return (
    <div className="banner warn with-action">
      <span>The daemon is not running, so {what}.</span>
      <button className="action" disabled={actions.busy} onClick={actions.startDaemon}>
        Start daemon
      </button>
    </div>
  );
}

/** Every server polled is unreachable in jump mode: most likely the jump host wants a login. */
export function needsConnect(gpus: GpuState): boolean {
  const polled = gpus.hosts.filter((h) => h.reachable !== null);
  return gpus.mode === "jump" && polled.length > 0 && polled.every((h) => !h.reachable);
}

/** What is wrong with the GPU data, if anything: no snapshot, an old one, or no way in. */
export function GpuBanners({ gpus, daemonPid, now }: { gpus: GpuState; daemonPid: number | null; now: number }) {
  const [connectError, setConnectError] = useState<string | null>(null);
  const out = [];
  if (daemonPid == null) {
    out.push(<DaemonBanner key="daemon" pid={daemonPid} what="GPU data is not refreshed" />);
  } else if (gpus.ts == null) {
    out.push(
      <div key="none" className="banner info">
        Waiting for the daemon's first GPU query (it polls every 30 s).
      </div>,
    );
  } else if (now - gpus.ts > STALE_SECONDS) {
    out.push(
      <div key="stale" className="banner warn">
        This GPU data is {fmtDuration(now - gpus.ts)} old. The daemon should refresh it within 30 s; if it doesn't,
        check <code>~/.gpu-spooler/daemon.log</code>.
      </div>,
    );
  }
  if (needsConnect(gpus)) {
    out.push(
      <div key="connect" className="banner error with-action">
        <span>
          No server could be reached through the jump host. If it asks for a password or one-time code, log in once
          with <code>task connect</code> and the connection stays open for the daemon.
          {connectError && <div className="small">Could not open Terminal: {connectError}</div>}
        </span>
        <button
          className="action"
          onClick={() =>
            commands.openTaskConnect().then((r) => setConnectError(r.status === "error" ? r.error : null))
          }
        >
          Open Terminal
        </button>
      </div>,
    );
  }
  return <>{out}</>;
}
