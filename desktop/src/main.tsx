import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { App } from "./App";
import { Popover } from "./Popover";
import "./styles.css";

async function start() {
  // `npm run dev` in a browser: no Tauri, so preview with made-up data (src/dev/mock.ts).
  if (import.meta.env.DEV && !("__TAURI_INTERNALS__" in window)) (await import("./dev/mock")).install();
  createRoot(document.getElementById("root")!).render(
    <StrictMode>{getCurrentWindow().label === "popover" ? <Popover /> : <App />}</StrictMode>,
  );
}

start();
