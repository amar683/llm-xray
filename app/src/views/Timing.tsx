import { useState } from "react";
import { pct, type Conn, type JobEvent, type TimingPhase, type TimingResult } from "../api";
import { Bars, Segmented, StackedBars, Tile } from "../components/charts";
import { RunPanel, useRuns } from "../components/RunPanel";
import { RuntimesPanel } from "../components/RuntimesPanel";
import type { Model } from "./Models";

const PARTS = [
  { key: "attention", label: "Attention", color: "var(--s1)" },
  { key: "mlp", label: "MLP", color: "var(--s2)" },
  { key: "lm_head", label: "Output head", color: "var(--s3)" },
  { key: "other", label: "Other", color: "var(--s-other)" },
];

const shareSegments = (p: TimingPhase) => {
  const s = p.shares;
  const values: Record<string, number> = {
    attention: s.attention,
    mlp: s.mlp,
    lm_head: s.lm_head,
    other: s.layer_other + s.embed + s.final_norm,
  };
  return PARTS.map((part) => ({ ...part, value: values[part.key] * p.wall_ms }));
};

export function Timing({ conn, model, events, openRunId }: { conn: Conn; model: Model; events: Record<string, JobEvent>; openRunId?: string | null }) {
  const [contexts, setContexts] = useState("128,512,2048");
  const runs = useRuns<TimingResult>(conn, model.path, "timing", events, openRunId);
  const [ctxIndex, setCtxIndex] = useState(0);
  const [phase, setPhase] = useState<"prefill" | "decode">("decode");
  const r = runs.detail?.result;
  const ctx = r?.contexts[Math.min(ctxIndex, r.contexts.length - 1)];
  const p = ctx?.[phase];

  return (
    <>
      <h1>Timing</h1>
      <p className="summary">Where the time goes when {model.id} reads a prompt (prefill) and generates each token (decode).</p>

      <RunPanel
        runs={runs.runs}
        currentId={runs.currentId}
        onSelect={runs.setSelected}
        active={runs.active}
        onStart={() => runs.start({ contexts: contexts.split(",").map((c) => parseInt(c, 10)).filter((c) => c > 0) })}
        onCancel={runs.cancel}
        error={runs.error}
        startLabel="Run timing"
      >
        <label>
          Context lengths (tokens)
          <input value={contexts} onChange={(e) => setContexts(e.target.value)} />
        </label>
      </RunPanel>

      {!r && !runs.loading && <p className="muted">No timing run for this model yet. It takes under a minute for a small model.</p>}

      {r && ctx && p && (
        <>
          <div className="tiles">
            <Tile label="Decode speed" value={`${r.contexts[0].decode.tokens_per_s.toFixed(1)} tok/s`} note={`${r.contexts[0].decode.wall_ms.toFixed(1)} ms per token`} />
            <Tile label="Prefill speed" value={`${Math.round(r.contexts[r.contexts.length - 1].prefill.tokens_per_s).toLocaleString()} tok/s`} note={`at ${r.contexts[r.contexts.length - 1].context.toLocaleString()} tokens`} />
            <Tile label="Decode vs. memory limit" value={`${r.bandwidth.times_above_floor.toFixed(2)}×`} note={`floor ${r.bandwidth.decode_floor_ms.toFixed(1)} ms at ${Math.round(r.bandwidth.peak_gb_per_s)} GB/s`} />
            <Tile label="CPU dispatch" value={pct(r.contexts[0].decode.dispatch_fraction, 0)} note="of decode time" />
          </div>
          {r.layer_devices.offloaded_layers.length > 0 && (
            <p className="error">Layers not on {r.layer_devices.main} (slow): {r.layer_devices.offloaded_layers.join(", ")}</p>
          )}

          <section className="panel">
            <h2>Share of time by part</h2>
            <StackedBars
              rows={r.contexts.flatMap((c) =>
                (["prefill", "decode"] as const).map((ph) => ({
                  label: `${c.context.toLocaleString()} · ${ph}`,
                  note: `${c[ph].wall_ms.toFixed(1)} ms`,
                  segments: shareSegments(c[ph]),
                })),
              )}
              format={(v, total) => `${v.toFixed(2)} ms (${pct(v / total)})`}
            />
            <p className="muted">
              Each generated token reads {r.bandwidth.weights_read_per_token_gb.toFixed(2)} GB of weights. PyTorch timings; llama.cpp and MLX will differ. The output head's prefill cost is
              inflated here because PyTorch computes it for every prompt position.
            </p>
          </section>

          <section className="panel">
            <div className="panel-head">
              <h2>Per layer (estimated ms)</h2>
              <div className="row">
                <Segmented label="Context" value={ctxIndex} onChange={setCtxIndex} options={r.contexts.map((c, i) => ({ value: i, label: c.context.toLocaleString() }))} />
                <Segmented label="Phase" value={phase} onChange={setPhase} options={[{ value: "prefill", label: "Prefill" }, { value: "decode", label: "Decode" }]} />
              </div>
            </div>
            <Bars
              unit="ms"
              legend={PARTS}
              rows={[
                ...p.layers.map((l) => ({
                  key: l.layer,
                  label: `Layer ${l.layer}`,
                  segments: [
                    { ...PARTS[0], value: l.attn_ms },
                    { ...PARTS[1], value: l.mlp_ms },
                    { ...PARTS[3], value: l.other_ms },
                  ],
                })),
                { key: "lm_head", label: "Output head", segments: [{ ...PARTS[2], value: p.other.lm_head.est_ms }] },
                { key: "embed", label: "Embedding", segments: [{ ...PARTS[3], value: p.other.embed.est_ms }] },
                { key: "final_norm", label: "Final norm", segments: [{ ...PARTS[3], value: p.other.final_norm.est_ms }] },
              ]}
            />
          </section>
        </>
      )}

      <RuntimesPanel conn={conn} modelPath={model.path} events={events} />
    </>
  );
}
