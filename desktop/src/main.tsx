import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { App } from "./App";
import { Popover } from "./Popover";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {getCurrentWindow().label === "popover" ? <Popover /> : <App />}
  </StrictMode>,
);
