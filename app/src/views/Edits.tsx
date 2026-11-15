import { useState } from "react";
import { api, gb, pct, useAsync, type Conn, type EditResult, type EditStage, type JobEvent, type Patch } from "../api";
import { Tile } from "../components/charts";
import { RunPanel, useRuns } from "../components/RunPanel";
import type { Model } from "./Models";

const EXAMPLE = "drop_layers layers=10,16\nprune_neurons keep=0.9";
const STAGE_NAMES: Record<string, string> = { base: "Base", edited: "Edited", healed: "Healed" };

export function Edits({
  conn,
  model,
  events,
  openRunId,
  draft,
}: {
  conn: Conn;
  model: Model;
  events: Record<string, JobEvent>;
  openRunId?: string | null;
  draft?: string | null;
}) {
  const [commands, setCommands] = useState(draft || EXAMPLE);
  const [heal, setHeal] = useState(false);
  const [save, setSave] = useState(false);
  const [runCode, setRunCode] = useState(false);
  const [name, setName] = useState("");
  const [parent, setParent] = useState("");
  const [patchError, setPatchError] = useState<string | null>(null);
  const patches = useAsync(() => api<Patch[]>(conn, `/patches?model=${encodeURIComponent(model.path)}`), [conn, model.path]);
  const runs = useRuns<EditResult>(conn, model.path, "edit", events, openRunId);
  const parentPatch = patches.data?.find((p) => p.id === parent);

  // Record the commands as a patch when they are named or stacked, then run everything it implies.
  const start = async () => {
    setPatchError(null);
    let effective = lines;
    let patchId: string | null = null;
    try {
      if (name.trim() || parent) {
        const patch = await api<Patch>(conn, "/patches", { body: { model: model.path, name: name.trim(), commands: lines, parent_id: parent || null } });
        effective = patch.effective_commands;
        patchId = patch.id;
        setName("");
        patches.reload();
      }
      await runs.start({ commands: effective, patch_id: patchId, heal: heal ? { steps: 150 } : null, save, run_code: runCode });
    } catch (e) {
      setPatchError((e as Error).message);
    }
  };
  const removePatch = async (id: string) => {
    setPatchError(null);
    try {
      await api(conn, `/patches/${id}`, { method: "DELETE" });
      if (parent === id) setParent("");
      patches.reload();
    } catch (e) {
      setPatchError((e as Error).message);
    }
  };
  const r = runs.detail?.result;
  const lines = commands.split("\n").map((l) => l.trim()).filter(Boolean);
  const stages = r ? Object.entries(r.stages) : [];
  const base = r?.stages.base;
  const last = stages.length ? stages[stages.length - 1][1] : undefined;

  const rows: { label: string; value: (s: EditStage) => string }[] = r
    ? [
        { label: "Parameters", value: (s) => `${(s.size.params / 1e9).toFixed(3)} B` },
        { label: "Size (half precision)", value: (s) => gb(s.size.fp16_bytes) },
        { label: "Layers / MLP width", value: (s) => `${s.size.layers} / ${s.size.mlp_width}` },
        ...r.tasks.map((t) => ({
          label: `${t[0].toUpperCase() + t.slice(1)} accuracy`,
          value: (s: EditStage) => `${pct(s.tasks[t].accuracy, 0)} ± ${pct(s.tasks[t].stderr, 0)}`,
        })),
        { label: "Drift from base (KL)", value: (s) => s.drift.kl.toFixed(4) },
        { label: "Top prediction unchanged", value: (s) => pct(s.drift.top1_agreement) },
        { label: "Looped / ended normally", value: (s) => `${s.health.looped} / ${s.health.ended} of ${s.health.prompts}` },
      ]
    : [];

  return (
    <>
      <h1>Edits</h1>
      <p className="summary">Change {model.id}'s weights with small commands and see what each change costs. The model on disk is never modified.</p>

      <RunPanel
        runs={runs.runs}
        currentId={runs.currentId}
        onSelect={runs.setSelected}
        active={runs.active}
        onStart={start}
        onCancel={runs.cancel}
        error={lines.length ? (patchError ?? runs.error) : "Enter at least one command."}
        startLabel="Apply and measure"
        describe={(run) => (run.params.commands as string[] | undefined)?.join("; ").slice(0, 44) ?? ""}
      >
        <label className="grow">
          Commands, one per line
          <textarea rows={4} value={commands} onChange={(e) => setCommands(e.target.value)} spellCheck={false} />
        </label>
        <div className="stack">
          <label>
            Build on
            <select value={parent} onChange={(e) => setParent(e.target.value)}>
              <option value="">Base model</option>
              {patches.data?.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} ({p.effective_commands.length} command{p.effective_commands.length === 1 ? "" : "s"})
                </option>
              ))}
            </select>
          </label>
          <label>
            Save as patch (name)
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder={parent ? "required when building on a patch" : "optional"} />
          </label>
        </div>
        <div className="stack">
          <label className="check" title="Short fine-tune that pulls the edited model back towards the base. Uses full precision, so it needs about twice the memory.">
            <input type="checkbox" checked={heal} onChange={(e) => setHeal(e.target.checked)} />
            Heal after editing
          </label>
          <label className="check">
            <input type="checkbox" checked={save} onChange={(e) => setSave(e.target.checked)} />
            Save the result as a model folder
          </label>
          <label className="check" title="Scores a small coding task by running the model's code in a sandbox with no network and no file writes.">
            <input type="checkbox" checked={runCode} onChange={(e) => setRunCode(e.target.checked)} />
            Also run the code task (sandboxed)
          </label>
        </div>
      </RunPanel>

      {parentPatch && (
        <section className="panel">
          <p>
            Building on <b>{parentPatch.chain.map((c) => c.name).join(" → ")}</b>. These commands run first:
          </p>
          <ul className="plain mono">
            {parentPatch.effective_commands.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </section>
      )}

      {!!patches.data?.length && (
        <section className="panel">
          <h2>Patch history</h2>
          <table>
            <thead>
              <tr>
                <th>Patch</th>
                <th>Builds on</th>
                <th>Its own commands</th>
                <th className="num">Total commands</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {patches.data.map((p) => (
                <tr key={p.id} className={p.id === parent ? "current" : ""}>
                  <td>{p.name}</td>
                  <td>{p.chain.length > 1 ? p.chain[p.chain.length - 2].name : "Base model"}</td>
                  <td className="mono">
                    {p.commands.map((c, i) => (
                      <div key={i}>{c}</div>
                    ))}
                  </td>
                  <td className="num">{p.effective_commands.length}</td>
                  <td className="num">
                    <button onClick={() => setParent(p.id)}>Build on</button>{" "}
                    <button onClick={() => removePatch(p.id)} disabled={patches.data!.some((q) => q.parent_id === p.id)} title="A patch with others built on it cannot be deleted.">
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted">A patch stores only its own commands. Running it applies everything it builds on first. The base model is never changed, so going back is choosing an earlier patch.</p>
        </section>
      )}

      <section className="panel">
        <details>
          <summary>Commands</summary>
          <table>
            <tbody>
              <tr>
                <td><code>zero_head layer=13 head=4</code></td>
                <td>Switch off one attention head.</td>
              </tr>
              <tr>
                <td><code>scale_neuron layer=12 neuron=2031 factor=0</code></td>
                <td>Scale one MLP neuron's output; 0 switches it off.</td>
              </tr>
              <tr>
                <td><code>drop_layers layers=10,16</code></td>
                <td>Remove whole layers. The model gets smaller and faster.</td>
              </tr>
              <tr>
                <td><code>prune_neurons keep=0.9</code></td>
                <td>Keep the most important share of MLP neurons in every layer.</td>
              </tr>
              <tr>
                <td><code>ablate_direction vector=FILE from_layer=2</code></td>
                <td>Stop layers from writing along a direction saved in a .npy file.</td>
              </tr>
            </tbody>
          </table>
        </details>
      </section>

      {!r && !runs.loading && <p className="muted">No edit run for this model yet. Without healing it takes about a minute; with healing, several.</p>}

      {r && base && last && (
        <>
          <div className="tiles">
            <Tile label="Parameters removed" value={pct(1 - last.size.params / base.size.params)} note={`${gb(base.size.fp16_bytes)} → ${gb(last.size.fp16_bytes)}`} />
            <Tile label="Top prediction unchanged" value={pct(last.drift.top1_agreement)} note="vs. the base model, on ordinary prompts" />
            {r.tasks.map((t) => (
              <Tile key={t} label={`${t[0].toUpperCase() + t.slice(1)} accuracy`} value={pct(last.tasks[t].accuracy, 0)} note={`base ${pct(base.tasks[t].accuracy, 0)}, ${base.tasks[t].n} questions`} />
            ))}
          </div>

          <section className="panel">
            <h2>What was applied</h2>
            <ul className="plain">
              {r.applied.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          </section>

          <section className="panel">
            <h2>Each stage against the base model</h2>
            <table>
              <thead>
                <tr>
                  <th />
                  {stages.map(([name]) => (
                    <th key={name} className="num">
                      {STAGE_NAMES[name] ?? name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.label}>
                    <td>{row.label}</td>
                    {stages.map(([name, s]) => (
                      <td key={name} className="num">
                        {row.value(s)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted">
              Accuracy is on questions held out from healing, ± one standard error. Drift compares next-token predictions with the base model on prompts not used for healing. Looped /
              ended counts come from generating on {base.health.prompts} chat prompts.
            </p>
            {r.healing && (
              <p className="muted">
                Healing: {r.healing.steps} steps on {r.healing.train_prompts} prompts, {(r.healing.trained_params / 1e6).toFixed(1)}M trained weights. Distance to the base on the training
                prompts went from {r.healing.loss_start.toFixed(3)} to {r.healing.loss_end.toFixed(3)}.
              </p>
            )}
            {r.code_isolation && <p className="muted">Code task isolation: {r.code_isolation}.</p>}
            {r.saved && (
              <p className="muted">
                Saved ({gb(r.saved.weights_bytes)} of weights): <code>{r.saved.path}</code>
              </p>
            )}
          </section>
        </>
      )}
    </>
  );
}
