import { useEffect, useState } from "react";
import { api, num, pct, useAsync, type Conn, type Heatmap, type JobEvent, type ProfileResult } from "../api";
import { Bars, HeatmapChart, Segmented, Tile } from "../components/charts";
import { RunPanel, useRuns } from "../components/RunPanel";
import type { Model } from "./Models";

type PromptSetInfo = { name: string; prompts: number; example: string };
type TopNeuron = { neuron: number; rate: number; rate_elsewhere: number };

export function Activations({ conn, model, events, openRunId }: { conn: Conn; model: Model; events: Record<string, JobEvent>; openRunId?: string | null }) {
  const available = useAsync(() => api<PromptSetInfo[]>(conn, "/prompt-sets"), [conn]);
  const [chosen, setChosen] = useState<string[] | null>(null);
  const [limit, setLimit] = useState(50);
  const [newTokens, setNewTokens] = useState(16);
  const runs = useRuns<ProfileResult>(conn, model.path, "profile", events, openRunId);
  const r = runs.detail?.result;
  const runId = runs.currentId;

  const setNames = r ? Object.keys(r.sets) : [];
  const [setName, setSetName] = useState<string>("");
  const [mode, setMode] = useState<"rate" | "specific">("rate");
  const [layer, setLayer] = useState<number | null>(null);
  const activeSet = setNames.includes(setName) ? setName : (setNames[0] ?? "");
  const canCompare = setNames.length > 1;
  const activeMode = canCompare ? mode : "rate";
  useEffect(() => setLayer(null), [runId]);

  const heat = useAsync(
    () => (r && runId && activeSet ? api<Heatmap>(conn, `/runs/${runId}/heatmap?set=${encodeURIComponent(activeSet)}&mode=${activeMode}&bins=240`) : Promise.resolve(null)),
    [conn, runId, activeSet, activeMode, !!r],
  );
  const top = useAsync(
    () => (r && runId && layer !== null ? api<TopNeuron[]>(conn, `/runs/${runId}/neurons?layer=${layer}&set=${encodeURIComponent(activeSet)}&top=8`) : Promise.resolve(null)),
    [conn, runId, layer, activeSet, !!r],
  );

  const selectedSets = chosen ?? available.data?.map((s) => s.name) ?? [];
  const toggle = (name: string) => setChosen(selectedSets.includes(name) ? selectedSets.filter((s) => s !== name) : [...selectedSets, name]);
  const red = r?.redundancy;

  return (
    <>
      <h1>Activations</h1>
      <p className="summary">Which neurons in {model.id} fire for which kind of prompt, and which layers it could do without.</p>

      <RunPanel
        runs={runs.runs}
        currentId={runs.currentId}
        onSelect={runs.setSelected}
        active={runs.active}
        onStart={() => runs.start({ sets: selectedSets, limit, new_tokens: newTokens })}
        onCancel={runs.cancel}
        error={selectedSets.length === 0 ? "Pick at least one prompt set." : runs.error}
        startLabel="Run profile"
      >
        <fieldset>
          <legend>Prompt sets</legend>
          {available.data?.map((s) => (
            <label key={s.name} className="check" title={s.example}>
              <input type="checkbox" checked={selectedSets.includes(s.name)} onChange={() => toggle(s.name)} />
              {s.name}
            </label>
          ))}
        </fieldset>
        <label>
          Prompts per set
          <input type="number" min={1} max={50} value={limit} onChange={(e) => setLimit(Number(e.target.value) || 1)} />
        </label>
        <label>
          Tokens to generate
          <input type="number" min={0} max={128} value={newTokens} onChange={(e) => setNewTokens(Number(e.target.value) || 0)} />
        </label>
      </RunPanel>

      {!r && !runs.loading && <p className="muted">No profile run for this model yet. 200 prompts take a few minutes on a small model.</p>}

      {r && (
        <>
          <div className="tiles">
            <Tile label="Prompts" value={num(r.run.prompts)} note={`${setNames.length} set${setNames.length === 1 ? "" : "s"}`} />
            <Tile label="Tokens measured" value={num(r.all.tokens)} note={`${num(r.run.generated_tokens)} generated`} />
            <Tile label="Run time" value={`${Math.round(r.run.seconds)} s`} />
            <Tile label="Peak memory" value={`${r.memory.peak_gb.toFixed(2)} GB`} note={`${r.memory.growth_gb >= 0 ? "+" : ""}${Math.round(r.memory.growth_gb * 1000)} MB over the run`} />
          </div>

          <section className="panel">
            <h2>By prompt set</h2>
            <table>
              <thead>
                <tr>
                  <th>Set</th>
                  <th className="num">Prompts</th>
                  <th className="num">Tokens</th>
                  <th className="num">Neurons firing</th>
                  <th className="num">Specific to this set</th>
                  <th className="num">Idle here, used elsewhere</th>
                </tr>
              </thead>
              <tbody>
                {setNames.map((name) => {
                  const s = r.sets[name];
                  const cmp = r.comparison[name];
                  const firing = s.layers.reduce((a, l) => a + l.mean_firing_rate, 0) / s.layers.length;
                  return (
                    <tr key={name}>
                      <td>{name}</td>
                      <td className="num">{s.prompts}</td>
                      <td className="num">{num(s.tokens)}</td>
                      <td className="num">{pct(firing)}</td>
                      <td className="num">{cmp ? pct(cmp.specific_fraction, 2) : "–"}</td>
                      <td className="num">{cmp ? pct(cmp.unused_fraction, 2) : "–"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="muted">
              Firing: activation above 0.1 in size. Specific: fires at least 20 points more often here than in the other sets. Idle here: fires on at most 1% of this set's tokens but at
              least 5% elsewhere.
            </p>
          </section>

          <section className="panel">
            <div className="panel-head">
              <h2>{activeMode === "rate" ? "How often neurons fire" : "Firing here minus elsewhere"}</h2>
              <div className="row">
                <Segmented label="Prompt set" value={activeSet} onChange={setSetName} options={setNames.map((n) => ({ value: n, label: n }))} />
                {canCompare && (
                  <Segmented label="Measure" value={mode} onChange={setMode} options={[{ value: "rate", label: "Firing rate" }, { value: "specific", label: "Vs. other sets" }]} />
                )}
              </div>
            </div>
            {heat.error && <p className="error">{heat.error}</p>}
            {heat.data && (
              <>
                <HeatmapChart
                  values={heat.data.values}
                  min={heat.data.min}
                  max={heat.data.max}
                  diverging={activeMode === "specific"}
                  selectedRow={layer}
                  onSelectRow={setLayer}
                  describe={(row, col, v) => (
                    <>
                      <b>Layer {row}</b>
                      <div>
                        Neurons ranked {Math.round((col / heat.data!.bins) * 100)}–{Math.round(((col + 1) / heat.data!.bins) * 100)}%
                      </div>
                      <div>{activeMode === "rate" ? `Fire on ${pct(v)} of tokens` : `${v >= 0 ? "+" : "−"}${pct(Math.abs(v))} vs. other sets`}</div>
                    </>
                  )}
                />
                <p className="muted">
                  One row per layer (0 at the top). Within each row the {num(heat.data.neurons)} neurons are sorted from highest to lowest, so the row shows the spread, not neuron positions.
                  Click a row to list that layer's most {activeSet}-specific neurons.
                </p>
              </>
            )}
            {layer !== null && top.data && (
              <>
                <h3>
                  Layer {layer}: neurons most specific to {activeSet}
                </h3>
                <table className="narrow">
                  <thead>
                    <tr>
                      <th className="num">Neuron</th>
                      <th className="num">Fires in {activeSet}</th>
                      <th className="num">Fires elsewhere</th>
                    </tr>
                  </thead>
                  <tbody>
                    {top.data.map((n) => (
                      <tr key={n.neuron}>
                        <td className="num">{n.neuron}</td>
                        <td className="num">{pct(n.rate)}</td>
                        <td className="num">{pct(n.rate_elsewhere)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
          </section>

          {red && (
            <section className="panel">
              <h2>What happens when one layer is skipped</h2>
              <Bars
                unit="KL"
                rows={red.layers.map((l) => ({
                  key: l.layer,
                  label: `Layer ${l.layer}`,
                  mark: red.most_removable.includes(l.layer) ? "least harmful" : undefined,
                  segments: [{ key: "kl", label: "Prediction shift (KL)", value: l.drop_kl, color: "var(--s1)" }],
                  tip: (
                    <>
                      <b>Layer {l.layer} skipped</b>
                      <div>Prediction shift (KL): {l.drop_kl.toFixed(3)}</div>
                      <div>Top prediction unchanged: {pct(l.drop_top1_agreement)}</div>
                      <div>Loss change: {l.drop_nll_increase >= 0 ? "+" : "−"}{Math.abs(l.drop_nll_increase).toFixed(3)}</div>
                      <div>Output vs. input similarity: {l.in_out_cosine.toFixed(3)}</div>
                    </>
                  ),
                }))}
              />
              <p className="muted">
                Longer bar = the model's predictions move further when that layer is removed. Measured on {red.prompts} prompts ({num(red.positions)} positions), so small differences are
                noise.
              </p>
            </section>
          )}

          {canCompare && (
            <section className="panel">
              <h2>Most set-specific neurons</h2>
              <table>
                <thead>
                  <tr>
                    <th>Set</th>
                    <th>Layer.neuron (fires here vs. elsewhere)</th>
                  </tr>
                </thead>
                <tbody>
                  {setNames.map((name) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>
                        {r.comparison[name]?.top_specific.slice(0, 5).map((n) => (
                          <span className="chip" key={`${n.layer}.${n.neuron}`}>
                            {n.layer}.{n.neuron} <span className="muted">{pct(n.rate, 0)} vs. {pct(n.rate_elsewhere, 0)}</span>
                          </span>
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}
        </>
      )}
    </>
  );
}
