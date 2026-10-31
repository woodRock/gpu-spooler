import { useEffect, useState } from "react";
import { commands } from "./bindings";
import { needsConnect } from "./Banners";
import { ACTIVE, fmtDuration, runtime } from "./format";
import type { Task } from "./types";
import { useGpuKeepAlive, useGpus, useNow, useQueue } from "./useQueue";

/** The menu bar popover: your tasks and the cluster at a glance. Clicking a task opens it in the
 * main window. */
export function Popover() {
  const queue = useQueue();
  const gpus = useGpus();
  const [visible, setVisible] = useState(true);
  const [login, setLogin] = useState<boolean | null>(null);
  useEffect(() => {
    const onFocus = () => setVisible(true);
    const onBlur = () => setVisible(false);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") commands.hidePopover();
      // ⌘N and ⌘O, as the buttons do.
      if (e.metaKey && e.key.toLowerCase() === "n") commands.openMain(null, true);
      if (e.metaKey && e.key.toLowerCase() === "o") commands.openMain(null, false);
    };
    window.addEventListener("focus", onFocus);
    window.addEventListener("blur", onBlur);
    window.addEventListener("keydown", onKey);
    commands.launchAtLogin().then((r) => r.status === "ok" && setLogin(r.data));
    return () => {
      window.removeEventListener("focus", onFocus);
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("keydown", onKey);
    };
  }, []);
  useGpuKeepAlive(visible);
  const now = useNow(visible);
  if (!queue || !gpus) return <div className="pop loading">Reading the queue…</div>;

  const active = queue.tasks
    .filter((t) => ACTIVE.includes(t.status))
    .sort((a, b) => (a.status === b.status ? b.id - a.id : a.status === "running" ? -1 : 1));
  const recent = queue.tasks.filter((t) => !ACTIVE.includes(t.status)).slice(0, 3);
  const up = gpus.hosts.filter((h) => h.role === "worker" && h.reachable);
  const polled = gpus.hosts.filter((h) => h.role === "worker" && h.reachable !== null);
  const cells = up.flatMap((h) => h.gpus);
  const free = cells.filter((g) => g.state === "free").length;
  const stale = gpus.ts == null || now - gpus.ts > 120;

  const toggleLogin = async () => {
    const next = !login;
    const r = await commands.setLaunchAtLogin(next);
    if (r.status === "ok") setLogin(next);
  };

  return (
    <div className="pop">
      <header className="pop-head">
        <strong>gpu-spooler</strong>
        <span className="muted small">
          <span className={`dot ${queue.daemon_pid != null ? "ok" : "warn"}`} />{" "}
          {queue.daemon_pid != null ? "daemon running" : "daemon stopped"}
        </span>
      </header>

      <div className="pop-stats">
        <div>
          <strong>{active.filter((t) => t.status === "running").length}</strong>
          <span>running</span>
        </div>
        <div>
          <strong>{active.filter((t) => t.status === "queued").length}</strong>
          <span>queued</span>
        </div>
        <div title={stale ? "The GPU data is old or missing" : undefined}>
          <strong className={stale ? "muted" : ""}>{gpus.ts == null ? "–" : free}</strong>
          <span>free GPUs</span>
        </div>
        <div>
          <strong className={needsConnect(gpus) ? "err" : ""}>
            {up.length}/{polled.length}
          </strong>
          <span>servers up</span>
        </div>
      </div>

      <section className="pop-list">
        {active.length === 0 ? (
          <div className="pop-empty muted">Nothing queued or running.</div>
        ) : (
          active.map((t) => <PopRow key={t.id} task={t} now={now} />)
        )}
        {recent.length > 0 && (
          <>
            <h3>Recent</h3>
            {recent.map((t) => (
              <PopRow key={t.id} task={t} now={now} />
            ))}
          </>
        )}
      </section>

      <footer className="pop-foot">
        <button className="btn primary small" onClick={() => commands.openMain(null, true)}>
          New task
        </button>
        <button className="btn small" onClick={() => commands.openMain(null, false)}>
          Open
        </button>
        <span className="spacer" />
        {login !== null && (
          <label className="check small muted">
            <input type="checkbox" checked={login} onChange={toggleLogin} /> Open at login
          </label>
        )}
        <button className="btn small" onClick={() => commands.quit()} title="Quit gpu-spooler (the daemon keeps running)">
          Quit
        </button>
      </footer>
    </div>
  );
}

function PopRow({ task: t, now }: { task: Task; now: number }) {
  const where = t.server ?? (t.host_spec || "any host");
  const exit = t.status === "failed" && t.exit_code != null ? ` ${t.exit_code}` : "";
  return (
    <button className="pop-row" onClick={() => commands.openMain(t.id, false)}>
      <span className={`status s-${t.status}`} title={t.status + exit} />
      <span className="pop-task">
        <span className="pop-name">{t.name || t.command}</span>
        <span className="muted small">
          #{t.id} · {where}
          {t.status === "failed" && <> · exit{exit}</>}
        </span>
      </span>
      <span className="muted small num">{fmtDuration(runtime(t, now))}</span>
    </button>
  );
}
