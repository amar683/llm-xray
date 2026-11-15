import { useState } from "react";
import { api, gb, useAsync, type Anatomy, type Conn, type ModelInfo } from "../api";
import { Tile } from "../components/charts";

export type Model = { id: string; path: string };

export function Models({ conn, model, onSelect }: { conn: Conn; model: Model | null; onSelect: (m: Model) => void }) {
  const system = useAsync(() => api<Record<string, unknown>>(conn, "/system"), [conn]);
  const models = useAsync(() => api<ModelInfo[]>(conn, "/models"), [conn]);
  const [path, setPath] = useState("");
  const [error, setError] = useState<string | null>(null);

  const openPath = async () => {
    setError(null);
    try {
      const a = await api<Anatomy>(conn, "/models/inspect", { body: { model: path.trim() } });
      onSelect({ id: path.trim().split("/").filter(Boolean).pop() ?? path, path: a.path });
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const s = system.data;
  return (
    <>
      <h1>Models</h1>
      <p className="summary">Pick a model that is already on this machine. Nothing is downloaded.</p>

      {s && (
        <div className="tiles">
          <Tile label="Runs on" value={String(s.default_device).toUpperCase()} note={String(s.gpu ?? s.cpu)} />
          <Tile label="Memory" value={`${s.ram_total_gb} GB`} note={`${s.ram_available_gb} GB free now`} />
          <Tile label="CPU cores" value={String(s.cpu_cores_physical)} note={`${s.os} ${s.machine}`} />
        </div>
      )}

      <section className="panel">
        <h2>In your Hugging Face cache</h2>
        {models.loading && <p className="muted">Looking…</p>}
        {models.error && <p className="error">{models.error}</p>}
        {models.data?.length === 0 && <p className="muted">No models with safetensors weights found in the cache.</p>}
        {!!models.data?.length && (
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Family</th>
                <th className="num">Layers</th>
                <th className="num">Size</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {models.data.map((m) => (
                <tr key={m.path} className={model?.path === m.path ? "current" : ""}>
                  <td>{m.id}</td>
                  <td>{m.model_type ?? "unknown"}</td>
                  <td className="num">{m.n_layers ?? "–"}</td>
                  <td className="num">{gb(m.size_bytes)}</td>
                  <td className="num">
                    {m.supported ? (
                      model?.path === m.path ? (
                        <span className="muted">Selected</span>
                      ) : (
                        <button onClick={() => onSelect({ id: m.id, path: m.path })}>Select</button>
                      )
                    ) : (
                      <span className="muted">Not supported yet</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="panel">
        <h2>Or open a model folder</h2>
        <form
          className="row"
          onSubmit={(e) => {
            e.preventDefault();
            if (path.trim()) openPath();
          }}
        >
          <input
            aria-label="Model folder path"
            placeholder="/path/to/model-folder (with config.json and .safetensors files)"
            value={path}
            onChange={(e) => setPath(e.target.value)}
          />
          <button type="submit" disabled={!path.trim()}>
            Open
          </button>
        </form>
        {error && <p className="error">{error}</p>}
      </section>
    </>
  );
}
