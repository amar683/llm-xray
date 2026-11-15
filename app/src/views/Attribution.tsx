import { useState } from "react";
import { type AttributionResult, type Conn, type JobEvent } from "../api";
import { HeatmapChart, Segmented, SignedBars } from "../components/charts";
import { PromptControls, show } from "../components/PromptControls";
import { RunPanel, useRuns } from "../components/RunPanel";
import type { Model } from "./Models";

const signed = (v: number, digits = 2) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;

export function Attribution({ conn, model, events, openRunId }: { conn: Conn; model: Model; events: Record<string, JobEvent>; openRunId?: string | null }) {
  const [prompt, setPrompt] = useState("The capital of France is");
  const [raw, setRaw] = useState(true);
  const runs = useRuns<AttributionResult>(conn, model.path, "attribution", events, openRunId);
  const [kind, setKind] = useState<"attn" | "mlp">("mlp");
  const [measure, setMeasure] = useState<"ablate" | "direct">("ablate");
  const r = runs.detail?.result;
  const headMax = r ? Math.max(...r.head_direct.flat().map(Math.abs), 1e-9) : 1;

  return (
    <>
      <h1>Attribution</h1>
      <p className="summary">Which parts of {model.id} produced its next token.</p>

      <RunPanel
        runs={runs.runs}
        currentId={runs.currentId}
        onSelect={runs.setSelected}
        active={runs.active}
        onStart={() => runs.start({ prompt, chat: !raw })}
        onCancel={runs.cancel}
        error={prompt.trim() ? runs.error : "Enter a prompt."}
        startLabel="Run attribution"
        describe={(run) => String(run.params.prompt ?? "").slice(0, 40)}
      >
        <PromptControls prompt={prompt} setPrompt={setPrompt} raw={raw} setRaw={setRaw} />
      </RunPanel>

      {!r && !runs.loading && <p className="muted">No attribution run for this model yet. It takes well under a minute.</p>}

      {r && (
        <>
          <section className="panel">
            <p>
              After “{r.prompt}” the model picks <b>“{show(r.target)}”</b> over the runner-up “{show(r.runner_up)}” with a logit gap of <b>{r.logit_diff.toFixed(2)}</b>. Everything below is
              measured on that gap. The direct contributions add up to {r.direct_total.toFixed(2)}.
            </p>
          </section>

          <section className="panel">
            <div className="panel-head">
              <h2>{measure === "ablate" ? "Change in the gap when a block is switched off" : "Each block's own push on the gap"}</h2>
              <div className="row">
                <Segmented label="Block" value={kind} onChange={setKind} options={[{ value: "mlp", label: "MLP" }, { value: "attn", label: "Attention" }]} />
                <Segmented label="Measure" value={measure} onChange={setMeasure} options={[{ value: "ablate", label: "Switched off" }, { value: "direct", label: "Direct push" }]} />
              </div>
            </div>
            <SignedBars
              rows={r.components.map((c) => ({
                key: c.layer,
                label: `Layer ${c.layer}`,
                value: c[`${kind}_${measure}`],
                tip: (
                  <>
                    <b>
                      Layer {c.layer} {kind === "mlp" ? "MLP" : "attention"}
                    </b>
                    <div>Direct push: {signed(c[`${kind}_direct`])}</div>
                    <div>Gap change if switched off: {signed(c[`${kind}_ablate`])}</div>
                  </>
                ),
              }))}
            />
            <p className="muted">
              {measure === "ablate"
                ? "A bar to the left means the gap shrinks without that block, so the block was helping the chosen token, including through later layers."
                : "A bar to the right means the block's own output pushes towards the chosen token; to the left, towards the runner-up."}
            </p>
          </section>

          <section className="panel">
            <h2>Attention heads: direct push on the gap</h2>
            <HeatmapChart
              values={r.head_direct}
              min={-headMax}
              max={headMax}
              diverging
              colLabels={r.head_direct[0].map((_, h) => `head ${h}`)}
              scaleLabels={[`−${headMax.toFixed(1)} (runner-up)`, `+${headMax.toFixed(1)} (chosen token)`]}
              describe={(row, col, v) => (
                <>
                  <b>
                    Layer {row}, head {col}
                  </b>
                  <div>Direct push: {signed(v)}</div>
                </>
              )}
            />
            <table className="narrow">
              <thead>
                <tr>
                  <th>Head</th>
                  <th className="num">Direct push</th>
                  <th className="num">Gap change if switched off</th>
                </tr>
              </thead>
              <tbody>
                {r.heads.slice(0, 8).map((h) => (
                  <tr key={`${h.layer}.${h.head}`}>
                    <td>
                      Layer {h.layer}, head {h.head}
                    </td>
                    <td className="num">{signed(h.direct)}</td>
                    <td className="num">{signed(h.ablate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section className="panel">
            <h2>MLP neurons with the largest effect</h2>
            <table className="narrow">
              <thead>
                <tr>
                  <th>Neuron</th>
                  <th className="num">Attribution</th>
                  <th className="num">Activation at the last token</th>
                </tr>
              </thead>
              <tbody>
                {r.neurons.slice(0, 12).map((n) => (
                  <tr key={`${n.layer}.${n.neuron}`}>
                    <td>
                      Layer {n.layer}, neuron {n.neuron}
                    </td>
                    <td className="num">{signed(n.attribution)}</td>
                    <td className="num">{signed(n.activation)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted">Attribution is gradient × activation summed over the prompt: an estimate of how much the gap depends on that neuron.</p>
          </section>
        </>
      )}
    </>
  );
}
