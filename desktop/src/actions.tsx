// The window's actions, each behind a confirmation, reporting what happened as a toast.
import { useState } from "react";
import { commands } from "./bindings";
import type { Task } from "./types";
import { useUi } from "./ui";

type Run = (fn: () => Promise<void>) => Promise<void>;

/** `busy` while an action runs (they can take a while: most reach a server). */
function useBusy(): [boolean, Run] {
  const [busy, setBusy] = useState(false);
  const run: Run = async (fn) => {
    setBusy(true);
    try {
      await fn();
    } finally {
      setBusy(false);
    }
  };
  return [busy, run];
}

export function useActions() {
  const { confirm, toast } = useUi();
  const [busy, run] = useBusy();
  const warn = (warnings: string[]) => warnings.forEach((w) => toast("warn", w));

  const kill = (t: Task) =>
    run(async () => {
      const queued = t.status === "queued";
      const ok = await confirm(
        queued
          ? {
              title: `Cancel task #${t.id}?`,
              body: "It hasn't started, so nothing is running yet.",
              confirm: "Cancel task",
              dismiss: "Keep task",
              danger: true,
            }
          : {
              title: `Kill task #${t.id}?`,
              body: (
                <>
                  Sends SIGTERM to its process group on <strong>{t.server}</strong>, then SIGKILL after 10 s if it is
                  still alive. It is marked killed and not retried.
                </>
              ),
              confirm: "Kill",
              danger: true,
            },
      );
      if (!ok) return;
      const r = await commands.killTask(t.id);
      if (r.status === "error") return toast("error", r.error);
      toast(r.data.confirmed && r.data.action !== "none" ? "ok" : "warn", r.data.message);
    });

  const resubmit = (t: Task, then?: (id: number) => void) =>
    run(async () => {
      const ok = await confirm({
        title: `Queue a copy of task #${t.id}?`,
        body: (
          <>
            <code className="block">{t.command}</code>
            Same GPUs, hosts and directory: {t.num_gpus === 0 ? "a CPU job" : `${t.num_gpus} GPU`}
            {t.host_spec ? ` on ${t.host_spec}` : " on any host"}, in <span className="mono">{t.working_dir}</span>.
          </>
        ),
        confirm: "Resubmit",
      });
      if (!ok) return;
      const r = await commands.resubmitTask(t.id);
      if (r.status === "error") return toast("error", r.error);
      toast("ok", `Queued task #${r.data.task_id}, a copy of #${t.id}.`);
      warn(r.data.warnings);
      then?.(r.data.task_id);
    });

  const clear = (finished: number) =>
    run(async () => {
      const ok = await confirm({
        title: `Clear ${finished} finished task${finished === 1 ? "" : "s"}?`,
        body: "Removes done, failed, killed, cancelled and lost tasks from the list, deletes their logs here and their job folders on the servers. This can't be undone.",
        confirm: "Clear",
        danger: true,
      });
      if (!ok) return;
      const r = await commands.clearFinished();
      if (r.status === "error") return toast("error", r.error);
      toast("ok", `Cleared ${r.data.tasks} task${r.data.tasks === 1 ? "" : "s"} and ${r.data.log_files} log file${r.data.log_files === 1 ? "" : "s"}.`);
      if (r.data.failed_hosts.length)
        toast("warn", `Could not remove job folders on ${r.data.failed_hosts.join(", ")} (unreachable).`);
    });

  const startDaemon = () =>
    run(async () => {
      const ok = await confirm({
        title: "Start the daemon?",
        body: "Runs the installed task binary's daemon in the background. It places queued tasks and keeps running after this app quits; stop it with task daemon stop.",
        confirm: "Start daemon",
      });
      if (!ok) return;
      const r = await commands.startDaemon();
      if (r.status === "error") return toast("error", r.error);
      toast("ok", r.data != null ? `Daemon started (PID ${r.data}).` : "The daemon was already running.");
    });

  return { busy, kill, resubmit, clear, startDaemon };
}
