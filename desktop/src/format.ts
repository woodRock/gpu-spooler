// The CLI's formats (crates/gpu-spooler/src/fmt.rs, core::runtime), so the app and `task list` agree.
import type { Task_Serialize as Task } from "./bindings";

export const ACTIVE = ["queued", "running"];

/** `45s`, `1m 05s`, `2h 03m`; `-` when unknown. */
export function fmtDuration(seconds: number | null): string {
  if (seconds === null || seconds < 0) return "-";
  const s = Math.floor(seconds);
  const pad = (n: number) => String(n).padStart(2, "0");
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${pad(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${pad(Math.floor((s % 3600) / 60))}m`;
}

/** `10-25 13:46`, local time; `-` when unset. */
export function fmtTs(ts: number | null | undefined): string {
  if (!ts) return "-";
  const d = new Date(ts * 1000);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** How long a task ran (so far, if running); null if it never started. */
export function runtime(t: Task, now: number): number | null {
  if (!t.started_at) return null;
  if (t.finished_at) return t.finished_at - t.started_at;
  return t.status === "running" ? now - t.started_at : null;
}
