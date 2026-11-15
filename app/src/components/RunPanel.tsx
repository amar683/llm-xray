// Shared "run a job, watch it, pick an earlier result" controls.
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, isActive, useAsync, type Conn, type JobEvent, type RunDetail, type RunRow } from "../api";

export function useRuns<R>(conn: Conn, modelPath: string, type: string, events: Record<string, JobEvent>, openRunId?: string | null) {
  const list = useAsync(() => api<RunRow[]>(conn, "/runs?limit=200"), [conn]);
  const runs = useMemo(
    () => (list.data ?? []).filter((r) => r.type === type && r.status === "done" && r.model_path === modelPath),
    [list.data, type, modelPath],
  );
  const [selected, setSelected] = useState<string | null>(openRunId ?? null);
  const [error, setError] = useState<string | null>(null);
  const mine = useRef<Set<string>>(new Set());

  const active = Object.values(events).find((e) => e.type === type && isActive(e));

  // When one of our jobs finishes, refresh the list and show it.
  const finished = Object.values(events).filter((e) => mine.current.has(e.run_id) && !isActive(e));
  const finishedKey = finished.map((e) => e.run_id + e.event).join(",");
  useEffect(() => {
    for (const e of finished) {
      mine.current.delete(e.run_id);
      if (e.event === "done") setSelected(e.run_id);
      if (e.event === "failed") setError(e.error ?? "The run failed.");
      list.reload();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finishedKey]);

  const currentId = selected && runs.some((r) => r.id === selected) ? selected : (runs[0]?.id ?? null);
  const detail = useAsync(
    () => (currentId ? api<RunDetail<R>>(conn, `/runs/${currentId}`) : Promise.resolve(null)),
    [conn, currentId],
  );

  const start = async (params: Record<string, unknown>) => {
    setError(null);
    try {
      const { run_id } = await api<{ run_id: string }>(conn, "/jobs", { body: { type, params: { model: modelPath, ...params } } });
      mine.current.add(run_id);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const cancel = () => active && api(conn, `/jobs/${active.run_id}`, { method: "DELETE" }).catch(() => undefined);

  return { runs, currentId, setSelected, active, start, cancel, error, detail: detail.data, loading: list.loading || detail.loading };
}

export function RunPanel({
  runs,
  currentId,
  onSelect,
  active,
  onStart,
  onCancel,
  error,
  startLabel,
  children,
  describe,
}: {
  runs: RunRow[];
  currentId: string | null;
  onSelect: (id: string) => void;
  active?: JobEvent;
  onStart: () => void;
  onCancel: () => void;
  error: string | null;
  startLabel: string;
  children?: ReactNode;
  describe?: (run: RunRow) => string;
}) {
  const progress = active?.total ? `${active.done ?? 0}/${active.total}` : "";
  return (
    <section className="panel run-panel">
      <div className="run-controls">{children}</div>
      <div className="run-actions">
        {active ? (
          <>
            <span className="progress" role="status">
              <span className="spinner" />
              {active.event === "queued" ? "Waiting for another run to finish" : `${active.stage ?? "Starting"} ${progress}`}
            </span>
            <button onClick={onCancel}>Cancel</button>
          </>
        ) : (
          <button className="primary" onClick={onStart}>
            {startLabel}
          </button>
        )}
        {runs.length > 0 && (
          <label className="run-select">
            Showing
            <select value={currentId ?? ""} onChange={(e) => onSelect(e.target.value)}>
              {runs.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.started_at.replace("T", " ").slice(0, 16)} · {describe ? describe(r) : r.backend?.replace("pytorch-", "")}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {error && <p className="error">{error}</p>}
    </section>
  );
}
