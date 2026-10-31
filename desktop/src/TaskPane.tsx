import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useActions } from "./actions";
import { lineText, parseAnsi, type Line } from "./ansi";
import { commands } from "./bindings";
import { ACTIVE } from "./format";
import { useKeys } from "./keys";
import type { LogChunk, Task } from "./types";

type Tab = "log" | "info";

/** Older output is dropped from the window past this (the file keeps it). */
const MAX_CHARS = 2_000_000;
/** The daemon copies new output every 5 s; asking more often than that only shortens the wait. */
const FOLLOW_MS = 1000;

export function TaskPane({
  task,
  onClose,
  onQueued,
}: {
  task: Task;
  onClose: () => void;
  /** A copy was queued: its id. */
  onQueued: (id: number) => void;
}) {
  const [tab, setTab] = useState<Tab>("log");
  const actions = useActions();
  // l and i switch tabs, as in `task tui`.
  useKeys(
    (e) => {
      if (e.key === "l") setTab("log");
      if (e.key === "i") setTab("info");
    },
    [],
  );
  const active = ACTIVE.includes(task.status);
  return (
    <section className="pane">
      <header className="pane-head">
        <div className="pane-title">
          <span className="muted">#{task.id}</span> {task.name || task.command}
        </div>
        {active && (
          <button className="btn small danger-text" disabled={actions.busy} onClick={() => actions.kill(task)} title="k">
            {task.status === "queued" ? "Cancel" : "Kill"}
          </button>
        )}
        <button className="btn small" disabled={actions.busy} onClick={() => actions.resubmit(task, onQueued)} title="r">
          Resubmit
        </button>
        <div className="segmented small" role="tablist">
          {(["log", "info"] as Tab[]).map((t) => (
            <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>
              {t === "log" ? "Log" : "Info"}
            </button>
          ))}
        </div>
        <button className="icon" onClick={onClose} title="Close (Esc)" aria-label="Close">
          ×
        </button>
      </header>
      {tab === "log" ? <LogView key={task.id} task={task} /> : <InfoView task={task} />}
    </section>
  );
}

function InfoView({ task }: { task: Task }) {
  const [rows, setRows] = useState<[string, string][] | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Re-read when the task changes state (server, start and end times fill in).
  useEffect(() => {
    let live = true;
    commands.taskDetails(task.id).then((r) => {
      if (!live) return;
      if (r.status === "ok") setRows(r.data);
      else setError(r.error);
    });
    return () => {
      live = false;
    };
  }, [task.id, task.status, task.server, task.finished_at]);
  if (error) return <div className="pane-body empty">{error}</div>;
  if (!rows) return <div className="pane-body" />;
  return (
    <div className="pane-body info">
      <dl>
        {rows.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd className={label === "Command" || label.endsWith("dir") || label === "Log file" ? "mono" : ""}>
              {value}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function LogView({ task }: { task: Task }) {
  const [text, setText] = useState("");
  const [chunk, setChunk] = useState<LogChunk | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [follow, setFollow] = useState(true);
  const [filter, setFilter] = useState("");
  const scroller = useRef<HTMLDivElement>(null);
  const offset = useRef(0);
  const active = ACTIVE.includes(task.status);
  // f follows the newest output (and jumps to it).
  useKeys(
    (e) => {
      if (e.key === "f") setFollow((on) => !on);
    },
    [],
  );

  const take = (c: LogChunk) => {
    offset.current = c.offset;
    setChunk(c);
    setText((cur) => {
      const next = c.reset ? c.text : cur + c.text;
      if (next.length <= MAX_CHARS) return next;
      const cut = next.indexOf("\n", next.length - MAX_CHARS);
      return next.slice(cut + 1);
    });
  };

  // First load, then follow while the task can still write.
  useEffect(() => {
    let live = true;
    commands.logTail(task.id).then((r) => {
      if (!live) return;
      if (r.status === "ok") take(r.data);
      else setError(r.error);
    });
    return () => {
      live = false;
    };
  }, [task.id]);
  useEffect(() => {
    if (!active && chunk && !chunk.missing) {
      // Once more after it ends, for the output copied with the final status.
      commands.logSince(task.id, offset.current).then((r) => r.status === "ok" && take(r.data));
      return;
    }
    if (!active) return;
    const id = setInterval(() => {
      commands.logSince(task.id, offset.current).then((r) => r.status === "ok" && take(r.data));
    }, FOLLOW_MS);
    return () => clearInterval(id);
  }, [task.id, active]);

  const lines = useMemo(() => {
    const parsed = parseAnsi(text.endsWith("\n") ? text.slice(0, -1) : text).lines;
    return text === "" ? [] : parsed;
  }, [text]);
  const shown = useMemo(() => {
    const q = filter.toLowerCase();
    return q ? lines.filter((l) => lineText(l).toLowerCase().includes(q)) : lines;
  }, [lines, filter]);

  useLayoutEffect(() => {
    const el = scroller.current;
    if (follow && el) el.scrollTop = el.scrollHeight;
  }, [shown, follow]);

  const onScroll = () => {
    const el = scroller.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    if (atBottom !== follow) setFollow(atBottom);
  };

  let note: string | null = null;
  if (error) note = error;
  else if (chunk?.missing)
    note =
      task.status === "queued"
        ? `No output yet: the task is waiting for ${task.num_gpus === 0 ? "a CPU slot" : "a GPU"}.`
        : task.started_at
          ? "No log for this task: it was removed."
          : "It never started, so it has no log.";
  else if (chunk && text === "") note = active ? "No output yet." : "The task wrote no output.";

  return (
    <div className="pane-body log">
      <div className="log-bar">
        <input
          className="search"
          type="search"
          placeholder="Filter lines…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        {filter && (
          <span className="muted small">
            {shown.length.toLocaleString()} of {lines.length.toLocaleString()} lines
          </span>
        )}
        {chunk?.truncated && <span className="muted small">Showing the end of the log</span>}
        <span className="spacer" />
        {active && (
          <span className="muted small live">
            <span className="dot ok" /> live
          </span>
        )}
        <button className={`toggle ${follow ? "on" : ""}`} onClick={() => setFollow(!follow)} title="Keep the newest output in view">
          Follow
        </button>
      </div>
      {note ? (
        <div className="empty">{note}</div>
      ) : (
        <div className="log-text" ref={scroller} onScroll={onScroll}>
          {shown.map((l, i) => (
            <LogLine key={i} line={l} />
          ))}
        </div>
      )}
      {chunk && !chunk.missing && <div className="log-path mono muted small">{chunk.path}</div>}
    </div>
  );
}

function LogLine({ line }: { line: Line }) {
  return (
    <div className="log-line">
      {line.length === 0
        ? "​"
        : line.map((s, i) => {
            const cls = [s.fg && `fg-${s.fg}`, s.bold && "b", s.dim && "d"].filter(Boolean).join(" ");
            return cls ? (
              <span key={i} className={cls}>
                {s.text}
              </span>
            ) : (
              s.text
            );
          })}
    </div>
  );
}
