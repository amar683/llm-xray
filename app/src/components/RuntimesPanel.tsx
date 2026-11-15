// PyTorch (the analysis backend) next to the runtimes people actually use.
import { useEffect, useState } from "react";
import { api, gb, isActive, useAsync, type Conn, type JobEvent, type Runtimes } from "../api";

const KEY = "xray.llamacpp";

export function RuntimesPanel({ conn, modelPath, events }: { conn: Conn; modelPath: string; events: Record<string, JobEvent> }) {
  const data = useAsync(() => api<Runtimes>(conn, "/models/runtimes", { body: { model: modelPath } }), [conn, modelPath]);
  const [folder, setFolder] = useState(() => {
    try {
      return localStorage.getItem(KEY) ?? "";
    } catch {
      return "";
    }
  });
  const [mlx, setMlx] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const all = Object.values(events).filter((e) => e.type === "bench");
  const active = all.find(isActive);
  // Reload the table whenever a bench job finishes; show why if it failed.
  const finished = all.filter((e) => !isActive(e)).map((e) => e.run_id + e.event).join(",");
  useEffect(() => {
    if (!finished) return;
    data.reload();
    const failed = all.find((e) => e.event === "failed");
    if (failed) setError(failed.error ?? "The measurement failed.");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finished]);

  const start = async () => {
    setError(null);
    try {
      localStorage.setItem(KEY, folder);
    } catch {
      /* not remembered */
    }
    try {
      await api(conn, "/jobs", { body: { type: "bench", params: { model: modelPath, llama_cpp: folder.trim() || null, mlx } } });
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const d = data.data;
  const top = Math.max(...(d?.rows.map((r) => r.decode_tps) ?? [1]), 1e-9);
  const needsFolder = !folder.trim() && !d?.llama_cpp_configured;

  return (
    <section className="panel">
      <h2>PyTorch vs. real runtimes</h2>
      <p className="muted">
        X-Ray analyses the model in PyTorch. This measures the same model in llama.cpp and MLX, which is what you would actually run.
      </p>
      {d && d.rows.length > 0 && (
        <table className="cost-table">
          <thead>
            <tr>
              <th>Runtime</th>
              <th>Variant</th>
              <th className="num">File size</th>
              <th className="num">Prefill (tokens/s)</th>
              <th>Decode (tokens/s)</th>
              <th className="num">vs. PyTorch</th>
            </tr>
          </thead>
          <tbody>
            {d.rows.map((r) => (
              <tr key={r.runtime + r.variant}>
                <td>{r.runtime}</td>
                <td>{r.variant}</td>
                <td className="num">{r.file_bytes ? gb(r.file_bytes) : "–"}</td>
                <td className="num">{Math.round(r.prefill_tps).toLocaleString()}</td>
                <td className="bar-cell">
                  <span className="cell-bar" style={{ width: `${Math.max(1, (r.decode_tps / top) * 100)}%`, background: r.runtime === "PyTorch" ? "var(--s-other)" : "var(--s1)" }} />
                  <span className="cell-value">{r.decode_tps.toFixed(1)}</span>
                </td>
                <td className="num">{r.vs_pytorch ? `${r.vs_pytorch.toFixed(2)}×` : "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {d?.bench && (
        <p className="muted">
          Real runtimes measured {d.bench.started_at.replace("T", " ").slice(0, 16)}: a {d.bench.n_prompt}-token prompt, then {d.bench.n_gen} generated tokens.
          {!d.has_timing && " Run timing above to add the PyTorch row."}
        </p>
      )}
      {d?.bench?.skipped.map((s) => (
        <p key={s.runtime} className="muted">
          {s.runtime} was not measured: {s.reason}
        </p>
      ))}
      {d && !d.bench && <p className="muted">No real-runtime measurement for this model yet.</p>}

      <div className="run-controls" style={{ marginTop: 12 }}>
        <label className="grow">
          llama.cpp folder (already built)
          <input value={folder} onChange={(e) => setFolder(e.target.value)} placeholder={d?.llama_cpp_configured ? "set in the engine's environment" : "/path/to/llama.cpp"} />
        </label>
        <label className="check">
          <input type="checkbox" checked={mlx} onChange={(e) => setMlx(e.target.checked)} />
          Also measure MLX
        </label>
      </div>
      <div className="run-actions" style={{ marginTop: 12 }}>
        {active ? (
          <span className="progress" role="status">
            <span className="spinner" />
            {active.event === "queued" ? "Waiting for another run to finish" : `${active.stage ?? "Starting"}${active.total ? ` ${active.done ?? 0}/${active.total}` : ""}`}
          </span>
        ) : (
          <button onClick={start} disabled={needsFolder && !mlx}>
            Measure real runtimes
          </button>
        )}
        {needsFolder && <span className="muted">Without a llama.cpp folder only MLX is measured.</span>}
      </div>
      {error && <p className="error">{error}</p>}
    </section>
  );
}
