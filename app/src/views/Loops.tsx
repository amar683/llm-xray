import { useState } from "react";
import { pct, type Conn, type JobEvent, type LoopsResult } from "../api";
import { HeatmapChart, Tile } from "../components/charts";
import { show } from "../components/PromptControls";
import { RunPanel, useRuns } from "../components/RunPanel";
import type { Model } from "./Models";

export function Loops({
  conn,
  model,
  events,
  openRunId,
  onEdit,
}: {
  conn: Conn;
  model: Model;
  events: Record<string, JobEvent>;
  openRunId?: string | null;
  onEdit: (command: string) => void;
}) {
  const [limit, setLimit] = useState(10);
  const [newTokens, setNewTokens] = useState(120);
  const [raw, setRaw] = useState(false);
  const runs = useRuns<LoopsResult>(conn, model.path, "loops", events, openRunId);
  const r = runs.detail?.result;
  const a = r?.analysis;
  const copyMax = a ? Math.max(...a.copy_attention.flat(), 1e-9) : 1;

  return (
    <>
      <h1>Loops</h1>
      <p className="summary">Which prompts make {model.id} repeat itself, what the loop looks like inside, and whether switching parts off stops it.</p>

      <RunPanel
        runs={runs.runs}
        currentId={runs.currentId}
        onSelect={runs.setSelected}
        active={runs.active}
        onStart={() => runs.start({ limit, new_tokens: newTokens, chat: !raw })}
        onCancel={runs.cancel}
        error={runs.error}
        startLabel="Hunt for loops"
      >
        <label>
          Prompts per set
          <input type="number" min={1} max={50} value={limit} onChange={(e) => setLimit(Number(e.target.value) || 1)} />
        </label>
        <label>
          Tokens to generate
          <input type="number" min={40} max={400} value={newTokens} onChange={(e) => setNewTokens(Number(e.target.value) || 40)} />
        </label>
        <label className="check">
          <input type="checkbox" checked={raw} onChange={(e) => setRaw(e.target.checked)} />
          Plain text (no chat template)
        </label>
      </RunPanel>

      {!r && !runs.loading && <p className="muted">No loop hunt for this model yet. 40 prompts take several minutes.</p>}

      {r && (
        <>
          <div className="tiles">
            <Tile label="Prompts" value={String(r.summary.prompts)} note={`greedy, up to ${r.new_tokens} tokens`} />
            <Tile label="Looped" value={`${r.summary.looped} (${pct(r.summary.loop_rate, 0)})`} />
            <Tile label="Ended normally" value={String(r.summary.ended_normally)} />
            <Tile label="Hit the limit, no loop" value={String(r.summary.hit_limit_without_loop)} />
          </div>

          <section className="panel">
            <h2>By prompt set</h2>
            <table className="narrow">
              <thead>
                <tr>
                  <th>Set</th>
                  <th className="num">Prompts</th>
                  <th className="num">Looped</th>
                  <th className="num">Ended normally</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(r.by_set).map(([name, s]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td className="num">{s.prompts}</td>
                    <td className="num">
                      {s.looped} ({pct(s.loop_rate, 0)})
                    </td>
                    <td className="num">{s.ended_normally}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          {!a && (
            <section className="panel">
              <p>No loops were found, so there is nothing to look inside.</p>
            </section>
          )}

          {a && (
            <>
              <section className="panel">
                <h2>Inside one loop</h2>
                <p>
                  Prompt ({a.set}): “{a.prompt}”. The output starts repeating at generated token <b>{a.onset}</b>, every <b>{a.period}</b> token{a.period === 1 ? "" : "s"}: “
                  {show(a.loop_text)}”.
                </p>
                <HeatmapChart
                  values={a.recurrence.map((row) => row.map((v) => Math.max(0, v)))}
                  min={0}
                  max={1}
                  diverging={false}
                  scaleLabels={["different", "identical"]}
                  describe={(row, col, v) => (
                    <>
                      <b>
                        Layer {row}, token {a.columns[col]} “{show(a.tokens[col])}”
                      </b>
                      <div>Similarity to {a.period} token{a.period === 1 ? "" : "s"} earlier: {v.toFixed(2)}</div>
                      <div>{a.columns[col] >= a.onset + a.period ? "Inside the loop" : a.columns[col] >= a.onset ? "First cycle" : "Before the loop"}</div>
                    </>
                  )}
                />
                <p className="muted">
                  One row per layer, one column per generated token, left to right. Each cell compares the layer's state with its state one period earlier. Stronger colour = more alike.
                  {a.locked_from_layer !== null ? ` Inside the loop every layer from ${a.locked_from_layer} on is at 0.9 or more.` : " No layer reaches 0.9 inside the loop."}
                </p>
              </section>

              <section className="panel">
                <h2>Copy heads</h2>
                <HeatmapChart
                  values={a.copy_attention}
                  min={0}
                  max={copyMax}
                  diverging={false}
                  colLabels={a.copy_attention[0].map((_, h) => `head ${h}`)}
                  scaleLabels={["0%", pct(copyMax, 0)]}
                  describe={(row, col, v) => (
                    <>
                      <b>
                        Layer {row}, head {col}
                      </b>
                      <div>{pct(v)} of its attention goes to the token after the previous copy</div>
                    </>
                  )}
                />
                <p className="muted">
                  How much of each head's attention, inside the loop, goes back to the token that followed the previous copy. Strongest: {" "}
                  {a.copy_heads.slice(0, 5).map((h) => `layer ${h.layer} head ${h.head} (${pct(h.attention, 0)})`).join(", ")}.
                </p>
              </section>

              <section className="panel">
                <h2>Blocks pushing the repeated token</h2>
                <table className="narrow">
                  <thead>
                    <tr>
                      <th>Block</th>
                      <th className="num">Push inside the loop</th>
                      <th className="num">Push before it</th>
                    </tr>
                  </thead>
                  <tbody>
                    {a.writers.slice(0, 6).map((w) => (
                      <tr key={`${w.layer}${w.kind}`}>
                        <td>
                          Layer {w.layer} {w.kind === "mlp" ? "MLP" : "attention"}
                        </td>
                        <td className="num">{w.in_loop.toFixed(1)}</td>
                        <td className="num">{w.before.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {a.direction && (
                  <p className="muted">
                    Loop direction (average looping state minus average normal state) is clearest at layer {a.direction.layer}, {a.direction.separation_sigma.toFixed(0)} standard deviations
                    from normal.{" "}
                    {a.direction.warning_tokens && a.direction.warning_tokens > 0
                      ? `It rises ${a.direction.warning_tokens} token(s) before the loop starts.`
                      : "It gives no early warning in this case."}
                  </p>
                )}
                {a.direction?.file && (
                  <p>
                    <button onClick={() => onEdit(`ablate_direction vector="${a.direction!.file}" from_layer=${a.direction!.layer}`)}>Remove this direction in Edits</button>{" "}
                    <span className="muted">Opens the Edits view with the command filled in, so its effect on accuracy can be measured.</span>
                  </p>
                )}
              </section>
            </>
          )}

          {r.causal && (
            <section className="panel">
              <h2>Does switching parts off stop the loops?</h2>
              <table>
                <thead>
                  <tr>
                    <th>Intervention</th>
                    <th className="num">Still loop</th>
                    <th className="num">Now end normally</th>
                    <th className="num">Healthy prompts: tokens changed</th>
                    <th className="num">Healthy prompts: now loop</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Nothing (baseline)</td>
                    <td className="num">
                      {r.causal.looping_prompts_tested}/{r.causal.looping_prompts_tested}
                    </td>
                    <td className="num">0</td>
                    <td className="num">0%</td>
                    <td className="num">0/{r.causal.healthy_prompts_tested}</td>
                  </tr>
                  {r.causal.tests.map((t) => (
                    <tr key={t.intervention}>
                      <td>{t.intervention[0].toUpperCase() + t.intervention.slice(1)}</td>
                      <td className="num">
                        {t.still_looping}/{r.causal!.looping_prompts_tested}
                      </td>
                      <td className="num">{t.now_end_normally}</td>
                      <td className="num">{pct(t.healthy_tokens_changed, 0)}</td>
                      <td className="num">
                        {t.healthy_now_loop}/{r.causal!.healthy_prompts_tested}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="muted">
                Tested on {r.causal.looping_prompts_tested} prompts that looped and {r.causal.healthy_prompts_tested} that ended normally. A good fix lowers the first column without changing
                the healthy prompts.
              </p>
            </section>
          )}

          <section className="panel">
            <details>
              <summary>All {r.cases.length} prompts</summary>
              <table>
                <thead>
                  <tr>
                    <th>Set</th>
                    <th>Prompt</th>
                    <th>Outcome</th>
                    <th>End of output</th>
                  </tr>
                </thead>
                <tbody>
                  {r.cases.map((c) => (
                    <tr key={c.index}>
                      <td>{c.set}</td>
                      <td>{c.prompt}</td>
                      <td>
                        {c.loop ? (
                          <span className="verdict bad">
                            <span aria-hidden>✕</span> loop from token {c.loop.onset}, every {c.loop.period}
                          </span>
                        ) : c.ended ? (
                          <span className="verdict good">
                            <span aria-hidden>✓</span> ended
                          </span>
                        ) : (
                          <span className="verdict warn">
                            <span aria-hidden>!</span> hit limit
                          </span>
                        )}
                      </td>
                      <td className="muted small">{show(c.tail.slice(-50))}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          </section>
        </>
      )}
    </>
  );
}
