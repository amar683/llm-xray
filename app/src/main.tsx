import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { uiLog } from "./api";

// A desktop window has no visible console: pass uncaught errors to the shell's log.
window.addEventListener("error", (e) => uiLog(`error: ${e.message} (${e.filename}:${e.lineno})`));
window.addEventListener("unhandledrejection", (e) => uiLog(`unhandled rejection: ${String(e.reason)}`));
uiLog("window loaded");

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
