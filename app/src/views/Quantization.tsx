import { useState } from "react";
import { pct, type Conn, type JobEvent, type QuantPlan, type QuantResult, type QuantTensor } from "../api";
import { HeatmapChart, Segmented, StackedBars } from "../components/charts";
import { RunPanel, useRuns } from "../components/RunPanel";
import type { Model } from "./Models";

const KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];
const KIND_LABELS = ["attn q", "attn k", "attn v", "attn out", "MLP gate", "MLP up", "MLP down"];
const gbs = (bytes: number) => `${(bytes / 2 ** 30).toFixed(2)} GB`;
const tensorLabel = (t: QuantTensor) => (t.layer === null ? "Embeddings" : `Layer ${t.layer} · ${KIND_LABELS[KINDS.indexOf(t.kind)] ?? t.kind}`);

function PlanSummary({ plan, bitsOptions }: { plan: QuantPlan; bitsOptions: number[] }) {
  const u = plan.uniform;
  const m = plan.mixed;
  const same = new Set(Object.values(m.assignment)).size === 1 && Object.values(m.assignment)[0] === plan.target_bits;
  const change = u.kl ? m.kl / u.kl - 1 : 0;
  return (
    <div className="plan">
      <h3>Same size as uniform {plan.target_bits}-bit ({gbs(u.bytes)})</h3>
      {same ? (
        <p className="muted">The plan is the same as uniform: {plan.target_bits} bits is the lowest width on offer, so no trade is possible.</p>
      ) : (
        <p>
          <span className={`verdict ${change < 0 ? "good" : "bad"}`}>
            <span aria-hidden>{change < 0 ? "✓" : "✕"}</span> Mixed plan is {change < 0 ? "better" : "worse"} than uniform
          </span>{" "}
          on held-out prompts: prediction shift {change >= 0 ? "+" : "−"}
          {pct(Math.abs(change), 0)}.
        </p>
      )}
      <table>
        <thead>
          <tr>
            <th>Plan</th>
            <th className="num">Size</th>
            <th className="num">Bits per weight</th>
            <th className="num">Prediction shift (KL)</th>
            <th className="num">Top prediction unchanged</th>
            <th className="num">Loss change</th>
          </tr>
        </thead>
        <tbody>
          {(
            [
              ["Uniform", u],
              ["Mixed", m],
            ] as const
          ).map(([label, v]) => (
            <tr key={label}>
              <td>{label}</td>
              <td className="num">{gbs(v.bytes)}</td>
              <td className="num">{v.avg_bpw.toFixed(2)}</td>
              <td className="num">{v.kl.toFixed(4)}</td>
              <td className="num">{pct(v.top1_agreement)}</td>
              <td className="num">
                {v.nll_increase >= 0 ? "+" : "−"}
                {Math.abs(v.nll_increase).toFixed(4)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {!same && (
        <StackedBars
          labels={false}
          rows={[
            {
              label: "Mixed plan weights",
              segments: bitsOptions.map((b, i) => ({ key: String(b), label: `${b}-bit`, value: m.params_by_bits[String(b)] ?? 0, color: `var(--o${i + 1})` })),
            },
          ]}
          format={(v, total) => `${pct(v / total)} of weights`}
        />
      )}
    </div>
  );
}

export function Quantization({ conn, model, events, openRunId }: { conn: Conn; model: Model; events: Record<string, JobEvent>; openRunId?: string | null }) {
  const [targets, setTargets] = useState("4,5");
  const [calibration, setCalibration] = useState(16);
  const runs = useRuns<QuantResult>(conn, model.path, "quant", events, openRunId);
  const [planIndex, setPlanIndex] = useState(0);
  const r = runs.detail?.result;

  const byLayer: Record<number, Record<string, QuantTensor>> = {};
  for (const t of r?.tensors ?? []) if (t.layer !== null) (byLayer[t.layer] ??= {})[t.kind] = t;
  const layerIds = Object.keys(byLayer).map(Number).sort((a, b) => a - b);
  const grid = layerIds.map((l) => KINDS.map((k) => byLayer[l][k]?.kl ?? 0));
  const maxKl = Math.max(...grid.flat(), 1e-9);
  const embedding = r?.tensors.find((t) => t.layer === null);
  const plan = r?.plans[Math.min(planIndex, (r?.plans.length ?? 1) - 1)];
  const bitsOptions = r?.settings.bits_options ?? [];
  const ranked = [...(r?.tensors ?? [])].sort((a, b) => b.kl - a.kl).slice(0, 10);

  return (
    <>
      <h1>Quantization</h1>
      <p className="summary">Which parts of {model.id} tolerate fewer bits, and a mixed-precision plan that spends bits where they matter.</p>

      <RunPanel
        runs={runs.runs}
        currentId={runs.currentId}
        onSelect={runs.setSelected}
        active={runs.active}
        onStart={() => runs.start({ target_bits: targets.split(",").map((b) => parseInt(b, 10)).filter((b) => b > 0), calibration_prompts: calibration })}
        onCancel={runs.cancel}
        error={runs.error}
        startLabel="Run quantization test"
      >
        <label>
          Plan sizes (as uniform bit widths)
          <input value={targets} onChange={(e) => setTargets(e.target.value)} />
        </label>
        <label>
          Calibration prompts
          <input type="number" min={4} max={64} value={calibration} onChange={(e) => setCalibration(Number(e.target.value) || 4)} />
        </label>
      </RunPanel>

      {!r && !runs.loading && <p className="muted">No quantization run for this model yet. It takes several minutes: every weight matrix is tested on its own.</p>}

      {r && (
        <>
          <section className="panel">
            <h2>Mixed precision vs. uniform at the same size</h2>
            {r.plans.map((p) => (
              <PlanSummary key={p.target_bits} plan={p} bitsOptions={bitsOptions} />
            ))}
            <p className="muted">
              Simulated in PyTorch with {r.settings.block}-value blocks, which approximates GGUF quantization but is not identical. Full precision is {gbs(r.full_precision_bytes)}. Sensitivity
              was measured on {r.settings.calibration_prompts} prompts; plans were checked on {r.settings.heldout_prompts} different ones.
            </p>
          </section>

          <section className="panel">
            <h2>Sensitivity: each weight matrix quantized alone to {r.settings.probe_bits} bits</h2>
            <HeatmapChart
              values={grid}
              min={0}
              max={maxKl}
              diverging={false}
              scale={Math.sqrt}
              colLabels={KIND_LABELS}
              scaleLabels={["no effect", `KL ${maxKl.toFixed(3)}`]}
              describe={(row, col, v) => {
                const t = byLayer[layerIds[row]][KINDS[col]];
                return (
                  <>
                    <b>
                      Layer {layerIds[row]} · {KIND_LABELS[col]}
                    </b>
                    <div>Prediction shift (KL): {v.toFixed(5)}</div>
                    {t && <div>Top prediction unchanged: {pct(t.top1_agreement)}</div>}
                    {t && <div>{(t.params / 1e6).toFixed(1)}M weights, largest is {t.outlier_ratio.toFixed(0)}× typical</div>}
                    {plan && t && <div>Plan: {plan.mixed.assignment[t.name]} bits</div>}
                  </>
                );
              }}
            />
            <p className="muted">
              Stronger colour = the model's predictions move more when only that matrix is quantized. Colour uses a square-root scale so small values stay visible.
              {embedding && ` Embeddings (not shown): KL ${embedding.kl.toFixed(4)}.`}
            </p>
          </section>

          {plan && (
            <section className="panel">
              <div className="panel-head">
                <h2>Bits per matrix in the mixed plan</h2>
                <Segmented label="Plan size" value={planIndex} onChange={setPlanIndex} options={r.plans.map((p, i) => ({ value: i, label: `${p.target_bits}-bit size` }))} />
              </div>
              <table className="bits-grid">
                <thead>
                  <tr>
                    <th>Layer</th>
                    {KIND_LABELS.map((k) => (
                      <th key={k} className="num">
                        {k}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {layerIds.map((l) => (
                    <tr key={l}>
                      <td>{l}</td>
                      {KINDS.map((k) => {
                        const bits = plan.mixed.assignment[byLayer[l][k]?.name];
                        return (
                          <td key={k} className={`num bits b${bitsOptions.indexOf(bits) + 1}`}>
                            {bits ?? "–"}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              {embedding && <p className="muted">Embeddings: {plan.mixed.assignment[embedding.name]} bits.</p>}
            </section>
          )}

          <section className="panel">
            <h2>Most sensitive matrices</h2>
            <table>
              <thead>
                <tr>
                  <th>Matrix</th>
                  <th className="num">Weights</th>
                  <th className="num">Prediction shift (KL)</th>
                  <th className="num">Top prediction unchanged</th>
                  <th className="num">Largest vs. typical weight</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((t) => (
                  <tr key={t.name}>
                    <td>{tensorLabel(t)}</td>
                    <td className="num">{(t.params / 1e6).toFixed(1)}M</td>
                    <td className="num">{t.kl.toFixed(5)}</td>
                    <td className="num">{pct(t.top1_agreement)}</td>
                    <td className="num">{t.outlier_ratio.toFixed(0)}×</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </>
      )}
    </>
  );
}
