import { useEffect } from "react";
import { Modal } from "./ui";

/** Whether a plain-key shortcut should leave this key alone: the user is typing, a dialog is
 * open, or a modifier is held (those belong to the system menu or the ⌘ shortcuts). */
export function ignoreKey(e: KeyboardEvent): boolean {
  const t = e.target;
  return (
    t instanceof HTMLInputElement ||
    t instanceof HTMLTextAreaElement ||
    t instanceof HTMLSelectElement ||
    !!document.querySelector(".overlay") ||
    e.metaKey ||
    e.ctrlKey ||
    e.altKey
  );
}

/** Run `on` for keys `ignoreKey` lets through, while mounted. */
export function useKeys(on: (e: KeyboardEvent) => void, deps: unknown[]) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!ignoreKey(e)) on(e);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

const GROUPS: [string, [string[], string][]][] = [
  [
    "Anywhere",
    [
      [["⌘", "1"], "Queue"],
      [["⌘", "2"], "GPUs"],
      [["⌘", "3"], "Hosts"],
      [["⌘", "N"], "New task"],
      [["?"], "These shortcuts"],
    ],
  ],
  [
    "Queue",
    [
      [["↑"], "Previous task"],
      [["↓"], "Next task"],
      [["⌘", "F"], "Filter tasks"],
      [["/"], "Filter tasks"],
      [["n"], "New task"],
      [["Esc"], "Close the task"],
    ],
  ],
  [
    "A task",
    [
      [["k"], "Kill or cancel"],
      [["r"], "Resubmit"],
      [["l"], "Log"],
      [["i"], "Info"],
      [["f"], "Follow the log"],
    ],
  ],
];

export function Shortcuts({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" || e.key === "?") onClose();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onClose]);
  return (
    <Modal onClose={onClose}>
      <h2>Keyboard shortcuts</h2>
      <div className="shortcuts">
        {GROUPS.map(([group, keys]) => (
          <div key={group} style={{ display: "contents" }}>
            <h3>{group}</h3>
            {keys.map(([combo, what]) => (
              <div key={combo.join("") + what}>
                <span>{what}</span>
                <span>
                  {combo.map((k) => (
                    <kbd key={k}>{k}</kbd>
                  ))}
                </span>
              </div>
            ))}
          </div>
        ))}
      </div>
      <div className="dialog-buttons">
        <button className="btn" onClick={onClose} autoFocus>
          Done
        </button>
      </div>
    </Modal>
  );
}
