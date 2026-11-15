import { api, useAsync, type Conn, type RunRow } from "../api";
import type { Model } from "./Models";

type Step = { view: string; title: string; what: string; run?: string; time?: string };

const STEPS: Step[] = [
  { view: "anatomy", title: "Look at its anatomy", what: "Layers, parameters, and whether it fits this machine. Instant: nothing is loaded." },
  { view: "timing", title: "Time it", what: "Where the time goes per layer, and how it compares with llama.cpp and MLX.", run: "timing", time: "under a minute" },
  { view: "activations", title: "Profile its activations", what: "Which neurons fire for which kind of prompt, and which layers it could do without.", run: "profile", time: "a few minutes" },
  { view: "quantization", title: "Test quantization", what: "Which weights tolerate fewer bits, and a mixed-precision plan checked against uniform.", run: "quant", time: "about 5 minutes" },
  { view: "loops", title: "Hunt for loops", what: "Which prompts make it repeat itself, and what drives the loop inside.", run: "loops", time: "several minutes" },
  { view: "edits", title: "Try an edit", what: "Switch off a head, drop a layer or prune neurons, and see what it costs. The model on disk is never changed.", run: "edit", time: "about a minute" },
  { view: "report", title: "Read the report", what: "Everything measured, condensed to a few thousand tokens. Optionally ask an AI advisor what to try next." },
];

export function Start({ conn, model, onGo }: { conn: Conn; model: Model | null; onGo: (view: string) => void }) {
  const runs = useAsync(() => api<RunRow[]>(conn, "/runs?limit=500"), [conn]);
  const done = new Set((runs.data ?? []).filter((r) => r.status === "done" && r.model_path === model?.path).map((r) => r.type));
  const next = model ? STEPS.find((s) => s.run && !done.has(s.run)) : undefined;

  return (
    <>
      <h1>Start here</h1>
      <p className="summary">LLM X-Ray measures what a local language model does inside. Everything runs on this machine; nothing is uploaded.</p>

      <section className="panel">
        <ol className="steps">
          <li className={model ? "done" : "next"}>
            <div className="step-mark" aria-hidden>
              {model ? "✓" : "1"}
            </div>
            <div className="step-body">
              <b>Choose a model</b>
              <p className="muted">{model ? `Using ${model.id}.` : "Pick one that is already on this machine. Nothing is downloaded."}</p>
            </div>
            <button className={model ? "" : "primary"} onClick={() => onGo("models")}>
              {model ? "Change" : "Choose a model"}
            </button>
          </li>
          {STEPS.map((s, i) => {
            const finished = !!s.run && done.has(s.run);
            const isNext = next?.view === s.view;
            return (
              <li key={s.view} className={finished ? "done" : isNext ? "next" : ""}>
                <div className="step-mark" aria-hidden>
                  {finished ? "✓" : i + 2}
                </div>
                <div className="step-body">
                  <b>{s.title}</b>
                  <p className="muted">
                    {s.what}
                    {s.time && !finished ? ` Takes ${s.time} for a small model.` : ""}
                    {finished ? " Done for this model." : ""}
                  </p>
                </div>
                <button className={isNext ? "primary" : ""} disabled={!model} onClick={() => onGo(s.view)}>
                  Open
                </button>
              </li>
            );
          })}
        </ol>
        <p className="muted">The steps can be done in any order. Each run is saved, and the Runs view lists them all.</p>
      </section>
    </>
  );
}
