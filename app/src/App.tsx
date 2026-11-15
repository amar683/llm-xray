import { useEffect, useState } from "react";
import { api, engineStatus, uiLog, inDesktopApp, isActive, loadConn, modelName, saveConn, useJobEvents, type Conn, type EngineStatus, type RunRow } from "./api";
import { Activations } from "./views/Activations";
import { Anatomy } from "./views/Anatomy";
import { Attribution } from "./views/Attribution";
import { CostValue } from "./views/CostValue";
import { Edits } from "./views/Edits";
import { LogitLens } from "./views/LogitLens";
import { Loops } from "./views/Loops";
import { Models, type Model } from "./views/Models";
import { Quantization } from "./views/Quantization";
import { Report } from "./views/Report";
import { Runs } from "./views/Runs";
import { Start } from "./views/Start";
import { Timing } from "./views/Timing";
import "./App.css";

type ViewId = "start" | "models" | "anatomy" | "timing" | "activations" | "quantization" | "cost-value" | "lens" | "attribution" | "loops" | "edits" | "report" | "runs";
const RUN_VIEWS: Record<string, ViewId> = { timing: "timing", profile: "activations", quant: "quantization", lens: "lens", attribution: "attribution", loops: "loops", edit: "edits" };
const VIEWS: { id: ViewId; label: string; needsModel: boolean }[] = [
  { id: "start", label: "Start here", needsModel: false },
  { id: "models", label: "Models", needsModel: false },
  { id: "anatomy", label: "Anatomy", needsModel: true },
  { id: "timing", label: "Timing", needsModel: true },
  { id: "activations", label: "Activations", needsModel: true },
  { id: "quantization", label: "Quantization", needsModel: true },
  { id: "cost-value", label: "Cost vs Value", needsModel: true },
  { id: "lens", label: "Logit Lens", needsModel: true },
  { id: "attribution", label: "Attribution", needsModel: true },
  { id: "loops", label: "Loops", needsModel: true },
  { id: "edits", label: "Edits", needsModel: true },
  { id: "report", label: "Report", needsModel: true },
  { id: "runs", label: "Runs", needsModel: false },
];

function Connect({ onConnect, problem }: { onConnect: (c: Conn) => void; problem: string | null }) {
  const [url, setUrl] = useState("http://127.0.0.1:8791");
  const [token, setToken] = useState("");
  return (
    <main className="connect">
      <h1>LLM X-Ray</h1>
      <p className="summary">Connect to the engine running on this machine.</p>
      <form
        className="panel"
        onSubmit={(e) => {
          e.preventDefault();
          onConnect({ url: url.replace(/\/$/, ""), token });
        }}
      >
        <label>
          Engine address
          <input value={url} onChange={(e) => setUrl(e.target.value)} />
        </label>
        <label>
          Token
          <input type="password" value={token} onChange={(e) => setToken(e.target.value)} />
        </label>
        <button className="primary" type="submit">
          Connect
        </button>
        {problem && <p className="error">{problem}</p>}
        <p className="muted">
          Start the engine with <code>uv run xray serve --port 8791</code>. It prints the token.
        </p>
      </form>
    </main>
  );
}

/** Shown in the desktop app while the shell sets up and starts the engine. */
function Booting({ status }: { status: EngineStatus | null }) {
  const failed = status?.phase === "failed";
  return (
    <main className="connect">
      <h1>LLM X-Ray</h1>
      {failed ? (
        <section className="panel">
          <p className="error">The engine could not start.</p>
          <pre className="report">{status.message}</pre>
          <p className="muted">Quit and reopen the app to try again.</p>
        </section>
      ) : (
        <section className="panel">
          <p className="progress" role="status">
            <span className="spinner" />
            {status?.message ?? "Starting."}
          </p>
          {status?.phase === "setup" && <p className="muted">This only happens in full the first time. You can leave the window open and come back.</p>}
        </section>
      )}
    </main>
  );
}

function loadModel(): Model | null {
  try {
    const saved = sessionStorage.getItem("xray.model");
    return saved ? (JSON.parse(saved) as Model) : null;
  } catch {
    return null;
  }
}

function App() {
  const desktop = inDesktopApp();
  const [boot, setBoot] = useState<EngineStatus | null>(null);
  const [conn, setConn] = useState<Conn | null>(() => (desktop ? null : loadConn()));
  const [checked, setChecked] = useState(false);

  // In the desktop app the shell starts the engine: wait for it to report ready.
  useEffect(() => {
    if (!desktop) return;
    let live = true;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const status = await engineStatus();
        if (!live) return;
        setBoot(status);
        if (status.phase === "ready" && status.url && status.token) {
          uiLog(`engine ready at ${status.url}`);
          setConn({ url: status.url, token: status.token });
          return;
        }
        if (status.phase === "failed") return;
      } catch (e) {
        if (live) setBoot({ phase: "failed", message: String(e), url: null, token: null });
        return;
      }
      timer = window.setTimeout(poll, 700);
    };
    poll();
    return () => {
      live = false;
      window.clearTimeout(timer);
    };
  }, [desktop]);
  const [problem, setProblem] = useState<string | null>(null);
  const [model, setModelState] = useState<Model | null>(loadModel);
  const [view, setView] = useState<ViewId>("start");
  const [openRun, setOpenRun] = useState<{ type: string; id: string } | null>(null);
  const [draft, setDraft] = useState<string | null>(null);
  const { events, online } = useJobEvents(checked ? conn : null);

  // Verify the address and token before showing the app.
  useEffect(() => {
    if (!conn) return;
    let live = true;
    api(conn, "/system").then(
      () => live && (uiLog("connected to the engine"), setChecked(true), setProblem(null)),
      (e: Error) => live && (uiLog(`cannot reach the engine: ${e.message}`), setChecked(false), setProblem(e.message === "Failed to fetch" ? "Could not reach the engine at that address." : e.message), setConn(null)),
    );
    return () => {
      live = false;
    };
  }, [conn]);

  const setModel = (m: Model) => {
    setModelState(m);
    try {
      sessionStorage.setItem("xray.model", JSON.stringify(m));
    } catch {
      /* not remembered */
    }
  };

  if (desktop && (!conn || !checked)) return <Booting status={problem ? { phase: "failed", message: problem, url: null, token: null } : boot} />;
  if (!conn || !checked) {
    return (
      <Connect
        problem={problem}
        onConnect={(c) => {
          saveConn(c);
          setConn(c);
        }}
      />
    );
  }

  const active = Object.values(events).filter(isActive);
  const go = (id: ViewId) => {
    setOpenRun(null);
    setView(id);
  };
  const open = (run: RunRow) => {
    if (!run.model_path) return;
    setModel({ id: modelName(run.model_path), path: run.model_path });
    setOpenRun({ type: run.type, id: run.id });
    setView(RUN_VIEWS[run.type] ?? "runs");
  };
  const current = VIEWS.find((v) => v.id === view)!;
  const blocked = current.needsModel && !model;

  return (
    <div className="shell">
      <nav className="sidebar" aria-label="Views">
        <div className="brand">LLM X-Ray</div>
        <div className="current-model">
          <div className="muted small">Model</div>
          {model ? <div title={model.path}>{model.id}</div> : <div className="muted">None selected</div>}
        </div>
        <ul>
          {VIEWS.map((v) => (
            <li key={v.id}>
              <button className={v.id === view ? "nav active" : "nav"} aria-current={v.id === view ? "page" : undefined} onClick={() => go(v.id)}>
                {v.label}
              </button>
            </li>
          ))}
        </ul>
        {active.map((e) => (
          <div className="job" key={e.run_id} role="status">
            <span className="spinner" />
            <span>
              {e.type}: {e.event === "queued" ? "queued" : (e.stage ?? "starting")}
              {e.total ? ` ${e.done ?? 0}/${e.total}` : ""}
            </span>
          </div>
        ))}
        <div className="engine-status" role="status">
          <span className={online ? "dot on" : "dot"} /> {online ? "Engine connected" : "Engine not responding"}
        </div>
      </nav>
      <main className="content">
        {blocked ? (
          <>
            <h1>{current.label}</h1>
            <p className="summary">Pick a model first.</p>
            <button className="primary" onClick={() => go("models")}>
              Choose a model
            </button>
          </>
        ) : view === "start" ? (
          <Start conn={conn} model={model} onGo={(v) => go(v as ViewId)} />
        ) : view === "models" ? (
          <Models
            conn={conn}
            model={model}
            onSelect={(m) => {
              setModel(m);
              go("anatomy");
            }}
          />
        ) : view === "anatomy" ? (
          <Anatomy conn={conn} model={model!} />
        ) : view === "timing" ? (
          <Timing key={model!.path} conn={conn} model={model!} events={events} openRunId={openRun?.type === "timing" ? openRun.id : null} />
        ) : view === "activations" ? (
          <Activations key={model!.path} conn={conn} model={model!} events={events} openRunId={openRun?.type === "profile" ? openRun.id : null} />
        ) : view === "quantization" ? (
          <Quantization key={model!.path} conn={conn} model={model!} events={events} openRunId={openRun?.type === "quant" ? openRun.id : null} />
        ) : view === "cost-value" ? (
          <CostValue conn={conn} model={model!} />
        ) : view === "lens" ? (
          <LogitLens key={model!.path} conn={conn} model={model!} events={events} openRunId={openRun?.type === "lens" ? openRun.id : null} />
        ) : view === "attribution" ? (
          <Attribution key={model!.path} conn={conn} model={model!} events={events} openRunId={openRun?.type === "attribution" ? openRun.id : null} />
        ) : view === "loops" ? (
          <Loops
            key={model!.path}
            conn={conn}
            model={model!}
            events={events}
            openRunId={openRun?.type === "loops" ? openRun.id : null}
            onEdit={(command) => {
              setDraft(command);
              setOpenRun(null);
              setView("edits");
            }}
          />
        ) : view === "report" ? (
          <Report
            conn={conn}
            model={model!}
            onEdit={(commands) => {
              setDraft(commands);
              setOpenRun(null);
              setView("edits");
            }}
          />
        ) : view === "edits" ? (
          <Edits key={model!.path + (draft ?? "")} conn={conn} model={model!} events={events} openRunId={openRun?.type === "edit" ? openRun.id : null} draft={draft} />
        ) : (
          <Runs conn={conn} events={events} onOpen={open} />
        )}
      </main>
    </div>
  );
}

export default App;
