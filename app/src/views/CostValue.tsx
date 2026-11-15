import { useState } from "react";
import { api, pct, useAsync, type Conn, type CostValue as CostValueData } from "../api";
import { Segmented, Tile } from "../components/charts";
import type { Model } from "./Models";

const NAMES: Record<string, string> = { timing: "Timing", profile: "Activations", quant: "Quantization" };

function BarCell({ value, max, digits }: { value?: number; max: number; digits: number }) {
  if (value === undefined) return <td className="num muted">–</td>;
  return (
    <td className="bar-cell">
      <span className="cell-bar" style={{ width: `${Math.max(1, (value / max) * 100)}%` }} />
      <span className="cell-value">{value.toFixed(digits)}</span>
    </td>
  );
}

export function CostValue({ conn, model }: { conn: Conn; model: Model }) {
  const data = useAsync(() => api<CostValueData>(conn, "/models/cost-value", { body: { model: model.path } }), [conn, model.path]);
  const [order, setOrder] = useState<"layer" | "skip">("layer");
  const d = data.data;
  if (data.error) return <p className="error">{data.error}</p>;
  if (!d) return <p className="muted">Collecting results…</p>;

  const rows = [...d.layers];
  if (order === "skip") rows.sort((a, b) => (a.skip_kl ?? Infinity) - (b.skip_kl ?? Infinity));
  const max = (key: "decode_ms" | "skip_kl" | "quant_kl") => Math.max(...d.layers.map((l) => l[key] ?? 0), 1e-9);
  const times = d.layers.map((l) => l.decode_ms).filter((v): v is number => v !== undefined);
  const withSkip = d.layers.filter((l) => l.skip_kl !== undefined);
  const cheapest = [...withSkip].sort((a, b) => a.skip_kl! - b.skip_kl!).slice(0, 5).map((l) => l.layer);
  const fragile = [...d.layers.filter((l) => l.quant_kl !== undefined)].sort((a, b) => b.quant_kl! - a.quant_kl!).slice(0, 5).map((l) => l.layer);
  const totalBytes = d.layers.reduce((a, l) => a + l.bytes, 0);

  return (
    <>
      <h1>Cost vs Value</h1>
      <p className="summary">What each layer of {model.id} costs, next to how much the model depends on it.</p>

      {d.missing.length > 0 && (
        <section className="panel">
          <p>
            Some columns are empty because this model has no {d.missing.map((m) => NAMES[m]).join(" or ")} run yet. Run {d.missing.length > 1 ? "them" : "it"} from{" "}
            {d.missing.length > 1 ? "those views" : "that view"} to fill them in.
          </p>
        </section>
      )}

      <div className="tiles">
        {times.length > 0 && (
          <Tile
            label="Time per layer (decode)"
            value={`${Math.min(...times).toFixed(2)}–${Math.max(...times).toFixed(2)} ms`}
            note={Math.max(...times) / Math.min(...times) < 1.15 ? "nearly the same for every layer" : "varies by layer"}
          />
        )}
        <Tile label="Size per layer" value={`${(d.layers[0].bytes / 2 ** 20).toFixed(0)} MB`} note={`${pct(d.layers[0].bytes / totalBytes)} of all layers each`} />
        {cheapest.length > 0 && <Tile label="Least harmful to skip" value={cheapest.join(", ")} note="layers, by prediction shift" />}
        {fragile.length > 0 && <Tile label="Most hurt by quantization" value={fragile.join(", ")} note={`layers, at ${d.quant_probe_bits} bits`} />}
      </div>

      <section className="panel">
        <div className="panel-head">
          <h2>Per layer</h2>
          <Segmented label="Order" value={order} onChange={setOrder} options={[{ value: "layer", label: "By layer" }, { value: "skip", label: "Least harmful to skip first" }]} />
        </div>
        <table className="cost-table">
          <thead>
            <tr>
              <th>Layer</th>
              <th className="num">Size</th>
              <th>Decode time (ms)</th>
              <th>Damage if skipped (KL)</th>
              <th>Damage if quantized (KL)</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((l) => (
              <tr key={l.layer}>
                <td>{l.layer}</td>
                <td className="num">{(l.bytes / 2 ** 20).toFixed(0)} MB</td>
                <BarCell value={l.decode_ms} max={max("decode_ms")} digits={2} />
                <BarCell value={l.skip_kl} max={max("skip_kl")} digits={3} />
                <BarCell value={l.quant_kl} max={max("quant_kl")} digits={4} />
                <td>
                  {cheapest.includes(l.layer) && <span className="tag">Least harmful to skip</span>}
                  {fragile.includes(l.layer) && <span className="tag">Keep more bits</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted">
          Cost is time and size; value is how far the model's predictions move when the layer is skipped, or when its weights are quantized
          {d.quant_probe_bits ? ` to ${d.quant_probe_bits} bits` : ""}. Bars are scaled within each column. From the most recent run of each kind for this model.
        </p>
      </section>
    </>
  );
}
