import { useEffect, useState } from "react";
import { api, isActive, modelName, useAsync, type CompareRow, type Comparison, type Conn, type JobEvent, type RunRow } from "../api";

function fmt(value: number | null, format: string, signed = false) {
  if (value === null) return "–";
  const sign = signed && value > 0 ? "+" : value < 0 ? "−" : "";
  const v = Math.abs(value);
  if (format.startsWith("pct")) return `${sign}${(v * 100).toFixed(format === "pct2" ? 2 : 1)}${signed ? " pts" : "%"}`;
  const digits = format === "int" ? 0 : Number(format.slice(3));
  return `${sign}${v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
}

const describe = (r: RunRow) => {
  const p = r.params;
  if (Array.isArray(p.commands)) return (p.commands as string[]).join("; ");
  if (typeof p.prompt === "string") return p.prompt;
  if (Array.isArray(p.contexts)) return `contexts ${(p.contexts as number[]).join(", ")}`;
  if (Array.isArray(p.target_bits)) return `sizes ${(p.target_bits as number[]).join(", ")}-bit`;
  if (p.limit) return `${p.limit} prompts per set`;
  return "";
};

function ComparePanel({ conn, a, b }: { conn: Conn; a: RunRow; b: RunRow }) {
  const cmp = useAsync(() => api<Comparison>(conn, `/runs/compare?a=${a.id}&b=${b.id}`), [conn, a.id, b.id]);
  const label = (r: RunRow) => `${r.started_at.replace("T", " ").slice(5, 16)}${describe(r) ? ` · ${describe(r).slice(0, 40)}` : ""}`;
  return (
    <section className="panel">
      <h2>Comparison of two {a.type} runs</h2>
      {cmp.error && <p className="error">{cmp.error}</p>}
      {cmp.data && (
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th className="num">A: {label(a)}</th>
              <th className="num">B: {label(b)}</th>
              <th className="num">Change (B − A)</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {cmp.data.rows.map((row: CompareRow) => (
              <tr key={row.label}>
                <td>{row.label}</td>
                <td className="num">{fmt(row.a, row.format)}</td>
                <td className="num">{fmt(row.b, row.format)}</td>
                <td className="num">{row.change === null ? "–" : fmt(row.change, row.format, true)}</td>
                <td>
                  {row.verdict === "better" && (
                    <span className="verdict good">
                      <span aria-hidden>▲</span> better
                    </span>
                  )}
                  {row.verdict === "worse" && (
                    <span className="verdict bad">
                      <span aria-hidden>▼</span> worse
                    </span>
                  )}
                  {row.verdict === "same" && <span className="muted">same</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="muted">A is the older run. “Better” and “worse” are only shown where one direction is clearly good, such as accuracy or drift.</p>
    </section>
  );
}

const ICON: Record<string, string> = { done: "✓", failed: "✕", cancelled: "–", running: "…", queued: "…" };
const CLASS: Record<string, string> = { done: "good", failed: "bad", cancelled: "muted", running: "info", queued: "info" };

export function Runs({ conn, events, onOpen }: { conn: Conn; events: Record<string, JobEvent>; onOpen: (run: RunRow) => void }) {
  const runs = useAsync(() => api<RunRow[]>(conn, "/runs?limit=200"), [conn]);
  // Refresh whenever a job changes state.
  const key = Object.values(events).map((e) => e.run_id + (isActive(e) ? e.event === "queued" : e.event)).join(",");
  useEffect(() => runs.reload(), [key]); // eslint-disable-line react-hooks/exhaustive-deps
  const [picked, setPicked] = useState<string[]>([]);
  const toggle = (id: string) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p.slice(-1), id]));
  const chosen = picked.map((id) => runs.data?.find((r) => r.id === id)).filter((r): r is RunRow => !!r);
  const pair = chosen.length === 2 ? [...chosen].sort((x, y) => x.id.localeCompare(y.id)) : null;

  return (
    <>
      <h1>Runs</h1>
      <p className="summary">Everything the engine has run on this machine, newest first. Tick two finished runs of the same kind to compare them.</p>
      {pair && pair[0].type === pair[1].type && <ComparePanel conn={conn} a={pair[0]} b={pair[1]} />}
      {pair && pair[0].type !== pair[1].type && (
        <section className="panel">
          <p>
            Those two runs are different kinds ({pair[0].type} and {pair[1].type}), so they have no numbers in common. Pick two of the same kind.
          </p>
        </section>
      )}
      <section className="panel">
        {runs.error && <p className="error">{runs.error}</p>}
        {runs.data?.length === 0 && <p className="muted">No runs yet. Pick a model, then run Timing or Activations.</p>}
        {!!runs.data?.length && (
          <table>
            <thead>
              <tr>
                <th>Compare</th>
                <th>Started</th>
                <th>Type</th>
                <th>Model</th>
                <th>Backend</th>
                <th>Status</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {runs.data.map((r) => (
                <tr key={r.id} className={picked.includes(r.id) ? "current" : ""}>
                  <td>
                    {r.status === "done" && (
                      <input type="checkbox" aria-label={`Compare run ${r.id}`} checked={picked.includes(r.id)} onChange={() => toggle(r.id)} />
                    )}
                  </td>
                  <td>{r.started_at.replace("T", " ").slice(0, 16)}</td>
                  <td>
                    {r.type}
                    {describe(r) && <div className="muted small">{describe(r).slice(0, 60)}</div>}
                  </td>
                  <td>{modelName(r.model_path ?? (r.params.model as string))}</td>
                  <td>{r.backend?.replace("pytorch-", "") ?? ""}</td>
                  <td>
                    <span className={`verdict ${CLASS[r.status] ?? ""}`}>
                      <span aria-hidden>{ICON[r.status]}</span> {r.status}
                    </span>
                    {r.error && <div className="muted small">{r.error}</div>}
                  </td>
                  <td className="num">{r.status === "done" && r.model_path && <button onClick={() => onOpen(r)}>Open</button>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </>
  );
}
