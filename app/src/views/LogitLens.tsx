import { useState } from "react";
import { pct, type Conn, type JobEvent, type LensResult } from "../api";
import { HeatmapChart, Tile } from "../components/charts";
import { PromptControls, show } from "../components/PromptControls";
import { RunPanel, useRuns } from "../components/RunPanel";
import type { Model } from "./Models";

export function LogitLens({ conn, model, events, openRunId }: { conn: Conn; model: Model; events: Record<string, JobEvent>; openRunId?: string | null }) {
  const [prompt, setPrompt] = useState("The capital of France is");
  const [raw, setRaw] = useState(true);
  const [newTokens, setNewTokens] = useState(12);
  const runs = useRuns<LensResult>(conn, model.path, "lens", events, openRunId);
  const r = runs.detail?.result;

  return (
    <>
      <h1>Logit Lens</h1>
      <p className="summary">At which layer {model.id} settles on each token it generates.</p>

      <RunPanel
        runs={runs.runs}
        currentId={runs.currentId}
        onSelect={runs.setSelected}
        active={runs.active}
        onStart={() => runs.start({ prompt, new_tokens: newTokens, chat: !raw })}
        onCancel={runs.cancel}
        error={prompt.trim() ? runs.error : "Enter a prompt."}
        startLabel="Run logit lens"
        describe={(run) => String(run.params.prompt ?? "").slice(0, 40)}
      >
        <PromptControls prompt={prompt} setPrompt={setPrompt} raw={raw} setRaw={setRaw} />
        <label>
          Tokens to generate
          <input type="number" min={1} max={64} value={newTokens} onChange={(e) => setNewTokens(Number(e.target.value) || 1)} />
        </label>
      </RunPanel>

      {!r && !runs.loading && <p className="muted">No logit-lens run for this model yet. It takes a few seconds.</p>}

      {r && (
        <>
          <div className="tiles">
            <Tile label="Prompt" value={r.prompt.length > 28 ? `${r.prompt.slice(0, 28)}…` : r.prompt} />
            <Tile label="Generated" value={r.generated_text.trim().length > 28 ? `${r.generated_text.trim().slice(0, 28)}…` : r.generated_text.trim()} />
            <Tile label="Decided at layer" value={r.mean_answer_layer.toFixed(1)} note={`on average, of ${r.prob.length} layers`} />
          </div>

          <section className="panel">
            <h2>How sure each layer is of the final token</h2>
            <HeatmapChart
              values={r.prob}
              min={0}
              max={1}
              diverging={false}
              colLabels={r.predicted_tokens.length <= 16 ? r.predicted_tokens.map(show) : undefined}
              scaleLabels={["0%", "100%"]}
              describe={(row, col, v) => (
                <>
                  <b>
                    Layer {row} → “{show(r.predicted_tokens[col])}”
                  </b>
                  <div>Probability here: {pct(v)}</div>
                  <div>This layer's own top guess: “{show(r.top[row][col])}”</div>
                  <div>Decided at layer {r.answer_layer[col]}</div>
                </>
              )}
            />
            <p className="muted">
              One column per generated token, one row per layer (0 at the top). Each cell is the probability the model would give that token if it stopped at that layer and went straight
              to the output. Stronger colour = more sure.
            </p>
          </section>

          <section className="panel">
            <h2>Per token</h2>
            <table className="narrow">
              <thead>
                <tr>
                  <th>After</th>
                  <th>Model wrote</th>
                  <th className="num">Decided at layer</th>
                  <th className="num">Final probability</th>
                </tr>
              </thead>
              <tbody>
                {r.predicted_tokens.map((t, i) => (
                  <tr key={i}>
                    <td>{show(r.input_tokens[i])}</td>
                    <td>{show(t)}</td>
                    <td className="num">{r.answer_layer[i]}</td>
                    <td className="num">{pct(r.prob[r.prob.length - 1][i])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted">“Decided at” is the layer from which the top guess equals the final token and no longer changes.</p>
          </section>
        </>
      )}
    </>
  );
}
