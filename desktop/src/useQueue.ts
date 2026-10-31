import { useEffect, useState } from "react";
// The backend sends the `_Serialize` shapes (every field present).
import { commands, events, type QueueState_Serialize as QueueState } from "./bindings";

/** The queue, kept current by the backend's watcher: null until the first read arrives. */
export function useQueue(): QueueState | null {
  const [state, setState] = useState<QueueState | null>(null);
  useEffect(() => {
    let live = true;
    // Listen first, then ask for the latest: no change can fall between the two.
    const unlisten = events.queueState.listen((e) => live && setState(e.payload));
    commands.queueState().then((s) => live && s && setState((cur) => cur ?? s));
    return () => {
      live = false;
      unlisten.then((f) => f());
    };
  }, []);
  return state;
}

/** Seconds since the epoch, ticking every second while `active`. */
export function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    if (!active) return;
    setNow(Date.now() / 1000);
    const id = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(id);
  }, [active]);
  return now;
}
