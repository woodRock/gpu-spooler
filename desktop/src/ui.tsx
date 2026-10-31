import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";

export type ConfirmOpts = {
  title: string;
  body?: ReactNode;
  confirm: string;
  /** The button that backs out (default "Cancel"). */
  dismiss?: string;
  /** Red confirm button: the action can't be undone. */
  danger?: boolean;
};

type ToastKind = "ok" | "warn" | "error";
type Toast = { id: number; kind: ToastKind; text: string };

type Ui = {
  confirm: (opts: ConfirmOpts) => Promise<boolean>;
  toast: (kind: ToastKind, text: string) => void;
};

const UiContext = createContext<Ui | null>(null);

export function useUi(): Ui {
  const ui = useContext(UiContext);
  if (!ui) throw new Error("useUi outside UiProvider");
  return ui;
}

/** Confirmation dialogs and toasts for the whole window. */
export function UiProvider({ children }: { children: ReactNode }) {
  const [dialog, setDialog] = useState<(ConfirmOpts & { resolve: (ok: boolean) => void }) | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const next = useRef(0);

  const confirm = useCallback(
    (opts: ConfirmOpts) => new Promise<boolean>((resolve) => setDialog({ ...opts, resolve })),
    [],
  );
  const toast = useCallback((kind: ToastKind, text: string) => {
    const id = next.current++;
    setToasts((t) => [...t, { id, kind, text }]);
    // Problems stay until dismissed; confirmations go by themselves.
    if (kind === "ok") setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5000);
  }, []);
  const close = (ok: boolean) => {
    dialog?.resolve(ok);
    setDialog(null);
  };

  return (
    <UiContext.Provider value={{ confirm, toast }}>
      {children}
      {dialog && <ConfirmDialog opts={dialog} onClose={close} />}
      <div className="toasts">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind}`}>
            <span>{t.text}</span>
            <button className="icon" aria-label="Dismiss" onClick={() => setToasts((all) => all.filter((x) => x.id !== t.id))}>
              ×
            </button>
          </div>
        ))}
      </div>
    </UiContext.Provider>
  );
}

function ConfirmDialog({ opts, onClose }: { opts: ConfirmOpts; onClose: (ok: boolean) => void }) {
  const confirmButton = useRef<HTMLButtonElement>(null);
  const dismissButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    // Return confirms what is focused: the safe choice when the action can't be undone.
    (opts.danger ? dismissButton : confirmButton).current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose(false);
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onClose, opts.danger]);
  return (
    <Modal onClose={() => onClose(false)} className="confirm">
      <h2>{opts.title}</h2>
      {opts.body && <div className="dialog-body">{opts.body}</div>}
      <div className="dialog-buttons">
        <button ref={dismissButton} className="btn" onClick={() => onClose(false)}>
          {opts.dismiss ?? "Cancel"}
        </button>
        <button ref={confirmButton} className={`btn ${opts.danger ? "danger" : "primary"}`} onClick={() => onClose(true)}>
          {opts.confirm}
        </button>
      </div>
    </Modal>
  );
}

export function Modal({ children, onClose, className = "" }: { children: ReactNode; onClose: () => void; className?: string }) {
  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`dialog ${className}`} role="dialog" aria-modal="true">
        {children}
      </div>
    </div>
  );
}
