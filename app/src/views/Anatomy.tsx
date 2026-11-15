import { api, gb, num, pct, useAsync, type Anatomy as AnatomyData, type Conn, type Fit, type FitRow } from "../api";
import { Bars, StackedBars, Tile } from "../components/charts";
import type { Model } from "./Models";

const COMPONENTS: { key: string; label: string; color: string }[] = [
  { key: "mlp", label: "MLP", color: "var(--s2)" },
  { key: "attention", label: "Attention", color: "var(--s1)" },
  { key: "embeddings", label: "Embeddings", color: "var(--s3)" },
];

const VERDICT_ICON: Record<string, string> = { fits: "✓", "fits if you free memory": "!", "too large": "✕" };
const VERDICT_CLASS: Record<string, string> = { fits: "good", "fits if you free memory": "warn", "too large": "bad" };

function FitTable({ title, rows, recommended }: { title: string; rows: FitRow[]; recommended: string | null }) {
  return (
    <div>
      <h3>{title}</h3>
      <table>
        <thead>
          <tr>
            <th>Option</th>
            <th className="num">Needs</th>
            <th>Verdict</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.option}>
              <td>
                {r.option}
                {r.option === recommended && <span className="tag">Recommended</span>}
              </td>
              <td className="num">{gb(r.need_bytes)}</td>
              <td>
                <span className={`verdict ${VERDICT_CLASS[r.verdict]}`}>
                  <span aria-hidden>{VERDICT_ICON[r.verdict]}</span> {r.verdict}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Anatomy({ conn, model }: { conn: Conn; model: Model }) {
  const anatomy = useAsync(() => api<AnatomyData>(conn, "/models/inspect", { body: { model: model.path } }), [conn, model.path]);
  const fit = useAsync(() => api<Fit>(conn, "/models/fit", { body: { model: model.path } }), [conn, model.path]);
  const a = anatomy.data;
  if (anatomy.error) return <p className="error">{anatomy.error}</p>;
  if (!a) return <p className="muted">Reading model…</p>;

  const arch = a.architecture;
  const known = COMPONENTS.reduce((sum, c) => sum + (a.by_component[c.key] ?? 0), 0);
  const segments = [
    ...COMPONENTS.map((c) => ({ ...c, value: a.by_component[c.key] ?? 0 })),
    { key: "other", label: "Other", color: "var(--s-other)", value: a.totals.params - known },
  ];
  const weights = Object.entries(a.memory_estimates.weights_bytes);

  return (
    <>
      <h1>Anatomy</h1>
      <p className="summary">What {model.id} is made of. Read from the file headers; the weights are not loaded.</p>

      {(!a.compatibility.ok || a.compatibility.warnings.length > 0) && (
        <section className={a.compatibility.ok ? "panel notice" : "panel notice bad"} role="alert">
          <b>{a.compatibility.ok ? "Worth knowing" : "This model cannot be analysed"}</b>
          <ul className="plain">
            {[...a.compatibility.problems, ...a.compatibility.warnings].map((text) => (
              <li key={text}>{text}</li>
            ))}
          </ul>
          {!a.compatibility.ok && <p className="muted">Its anatomy is shown below, but every analysis that runs the model will refuse it.</p>}
        </section>
      )}

      <div className="tiles">
        <Tile label="Parameters" value={`${(a.totals.params / 1e9).toFixed(2)} B`} note={`${gb(a.totals.bytes_on_disk)} on disk`} />
        <Tile label="Layers" value={String(arch.n_layers)} note={`${num(a.layers[0].params)} params each`} />
        <Tile label="Hidden / MLP width" value={`${arch.hidden_size} / ${arch.mlp_size}`} />
        <Tile label="Attention heads" value={String(arch.n_heads)} note={`${arch.n_kv_heads} key/value heads`} />
        <Tile label="Vocabulary" value={num(Number(arch.vocab_size))} note={arch.tied_embeddings ? "shared with output head" : "separate output head"} />
        <Tile label="Max context" value={num(Number(arch.max_context))} note="tokens" />
      </div>

      <section className="panel">
        <h2>Where the parameters are</h2>
        <StackedBars
          rows={[{ label: "Parameters", segments }]}
          format={(v, total) => `${num(v)} parameters (${pct(v / total)})`}
        />
      </section>

      <section className="panel">
        <h2>Estimated weight size by quantization</h2>
        <Bars
          unit="GB"
          rows={weights.map(([q, bytes]) => ({
            key: q,
            label: q,
            segments: [{ key: "w", label: "Weights", value: bytes / 2 ** 30, color: "var(--s1)" }],
          }))}
        />
        <p className="muted">Approximate, from typical bits per weight for each GGUF type.</p>
      </section>

      <section className="panel">
        <h2>Does it fit this machine?</h2>
        {fit.error && <p className="error">{fit.error}</p>}
        {fit.data && (
          <>
            <p>
              {fit.data.device.toUpperCase()}: {gb(fit.data.available_bytes)} available now, {gb(fit.data.usable_bytes)} usable at most.
            </p>
            <div className="two-col">
              <FitTable title="Analysing here (PyTorch)" rows={fit.data.analysis} recommended={fit.data.recommended.analysis} />
              <FitTable
                title={`Running day to day (GGUF, ${num(fit.data.context)} context)`}
                rows={fit.data.running}
                recommended={fit.data.recommended.running}
              />
            </div>
            <p className="muted">Estimates, not measurements.</p>
          </>
        )}
      </section>
    </>
  );
}
