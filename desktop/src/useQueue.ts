import { useEffect, useState } from "react";
import { commands, events } from "./bindings";
import type { GpuState, QueueState } from "./types";

type Live<T> = {
  listen: (cb: (e: { payload: T }) => void) => Promise<() => void>;
};

/** A state the backend's watcher keeps current: null until the first read arrives. */
function useLive<T>(event: Live<T>, latest: () => Promise<T | null>): T | null {
  const [state, setState] = useState<T | null>(null);
  useEffect(() => {
    let live = true;
    // Listen first, then ask for the latest: no change can fall between the two.
    const unlisten = event.listen((e) => live && setState(e.payload));
    latest().then((s) => live && s && setState((cur) => cur ?? s));
    return () => {
      live = false;
      unlisten.then((f) => f());
    };
  }, [event, latest]);
  return state;
}

export function useQueue(): QueueState | null {
  return useLive<QueueState>(events.queueState, commands.queueState);
}

export function useGpus(): GpuState | null {
  return useLive<GpuState>(events.gpuState, commands.gpuState);
}

/** Seconds since the epoch, ticking every `everyMs` while `active`. */
export function useNow(active: boolean, everyMs = 1000): number {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    if (!active) return;
    setNow(Date.now() / 1000);
    const id = setInterval(() => setNow(Date.now() / 1000), everyMs);
    return () => clearInterval(id);
  }, [active, everyMs]);
  return now;
}

/** The daemon only refreshes the GPU snapshot while it is asked to (for two minutes after each
 * ask), so ask now and every minute while `active` and the window is visible. */
export function useGpuKeepAlive(active: boolean) {
  useEffect(() => {
    if (!active) return;
    const ask = () => document.visibilityState === "visible" && commands.requestGpuSnapshot();
    ask();
    const id = setInterval(ask, 60_000);
    document.addEventListener("visibilitychange", ask);
    return () => {
      clearInterval(id);
      document.removeEventListener("visibilitychange", ask);
    };
  }, [active]);
}
