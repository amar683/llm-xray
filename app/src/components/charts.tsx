// Small chart components. Colours come from CSS variables so light and dark
// themes are each chosen, not flipped. Text always uses text tokens; colour
// only marks identity or magnitude.
import { useEffect, useRef, useState, type ReactNode } from "react";

export type Segment = { key: string; label: string; value: number; color: string };

/** Hover tooltip that follows the pointer inside its positioned parent. */
export function useTooltip() {
  const [tip, setTip] = useState<{ x: number; y: number; content: ReactNode } | null>(null);
  const show = (e: React.MouseEvent, content: ReactNode) => {
    const box = (e.currentTarget as HTMLElement).closest(".chart")?.getBoundingClientRect();
    if (box) setTip({ x: e.clientX - box.left, y: e.clientY - box.top, content });
  };
  const hide = () => setTip(null);
  const node = tip && (
    <div className="tooltip" style={{ left: tip.x, top: tip.y }} role="tooltip">
      {tip.content}
    </div>
  );
  return { show, hide, node };
}

export function Legend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <ul className="legend">
      {items.map((i) => (
        <li key={i.label}>
          <span className="swatch" style={{ background: i.color }} />
          {i.label}
        </li>
      ))}
    </ul>
  );
}

/** One or more 100%-stacked horizontal bars sharing a legend. */
export function StackedBars({
  rows,
  format,
  labels = true,
}: {
  rows: { label: string; note?: string; segments: Segment[] }[];
  format: (value: number, total: number) => string;
  labels?: boolean;
}) {
  const { show, hide, node } = useTooltip();
  return (
    <div className="chart">
      <Legend items={rows[0].segments.map((s) => ({ label: s.label, color: s.color }))} />
      <div className="stack-rows">
        {rows.map((row) => {
          const total = row.segments.reduce((a, s) => a + s.value, 0) || 1;
          return (
            <div className="stack-row" key={row.label}>
              <div className="stack-label">{row.label}</div>
              <div className="stack-bar">
                {row.segments
                  .filter((s) => s.value > 0)
                  .map((s) => (
                    <div
                      key={s.key}
                      className="stack-seg"
                      style={{ flexGrow: s.value / total, background: s.color }}
                      onMouseMove={(e) =>
                        show(
                          e,
                          <>
                            <b>{s.label}</b>
                            <div>{format(s.value, total)}</div>
                          </>,
                        )
                      }
                      onMouseLeave={hide}
                    >
                      {labels && s.value / total > 0.09 && <span>{((s.value / total) * 100).toFixed(0)}%</span>}
                    </div>
                  ))}
              </div>
              {row.note && <div className="stack-note">{row.note}</div>}
            </div>
          );
        })}
      </div>
      {node}
    </div>
  );
}

/** Horizontal bars, one per row, optionally split into stacked segments. */
export function Bars({
  rows,
  max,
  unit,
  legend,
  onSelect,
  selected,
}: {
  rows: { key: string | number; label: string; segments: Segment[]; tip?: ReactNode; mark?: string }[];
  max?: number;
  unit: string;
  legend?: { label: string; color: string }[];
  onSelect?: (key: string | number) => void;
  selected?: string | number | null;
}) {
  const { show, hide, node } = useTooltip();
  const totals = rows.map((r) => r.segments.reduce((a, s) => a + s.value, 0));
  const top = max ?? Math.max(...totals, 1e-9);
  return (
    <div className="chart">
      {legend && <Legend items={legend} />}
      <div className="bars">
        {rows.map((row, i) => (
          <div
            key={row.key}
            className={`bar-row${onSelect ? " clickable" : ""}${selected === row.key ? " selected" : ""}`}
            onMouseMove={(e) =>
              show(
                e,
                row.tip ?? (
                  <>
                    <b>{row.label}</b>
                    {row.segments.map((s) => (
                      <div key={s.key}>
                        {s.label}: {s.value.toFixed(3)} {unit}
                      </div>
                    ))}
                  </>
                ),
              )
            }
            onMouseLeave={hide}
            onClick={() => onSelect?.(row.key)}
          >
            <div className="bar-label">{row.label}</div>
            <div className="bar-track">
              {row.segments
                .filter((s) => s.value > 0)
                .map((s) => (
                  <div key={s.key} className="bar-seg" style={{ width: `${(s.value / top) * 100}%`, background: s.color }} />
                ))}
            </div>
            <div className="bar-value">
              {totals[i].toFixed(totals[i] >= 100 ? 0 : 2)}
              {row.mark && <span className="bar-mark"> {row.mark}</span>}
            </div>
          </div>
        ))}
      </div>
      {node}
    </div>
  );
}

/** Horizontal bars for signed values, growing left or right from a centre line. */
export function SignedBars({ rows, digits = 2 }: { rows: { key: string | number; label: string; value: number; tip?: ReactNode }[]; digits?: number }) {
  const { show, hide, node } = useTooltip();
  const bound = Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);
  return (
    <div className="chart">
      <div className="bars signed">
        {rows.map((row) => {
          const width = `${(Math.abs(row.value) / bound) * 50}%`;
          return (
            <div
              key={row.key}
              className="bar-row"
              onMouseMove={(e) =>
                show(
                  e,
                  row.tip ?? (
                    <>
                      <b>{row.label}</b>
                      <div>{row.value.toFixed(3)}</div>
                    </>
                  ),
                )
              }
              onMouseLeave={hide}
            >
              <div className="bar-label">{row.label}</div>
              <div className="signed-track">
                <div className={row.value >= 0 ? "signed-bar pos" : "signed-bar neg"} style={row.value >= 0 ? { left: "50%", width } : { right: "50%", width }} />
              </div>
              <div className="bar-value">
                {row.value >= 0 ? "+" : "−"}
                {Math.abs(row.value).toFixed(digits)}
              </div>
            </div>
          );
        })}
      </div>
      {node}
    </div>
  );
}

function cssColor(el: Element, name: string): [number, number, number] {
  const hex = getComputedStyle(el).getPropertyValue(name).trim().replace("#", "");
  return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16)) as [number, number, number];
}
const mix = (a: number[], b: number[], t: number) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

/** Layers x columns grid on a canvas. Sequential (one hue) or diverging (two hues, neutral middle). */
export function HeatmapChart({
  values,
  min,
  max,
  diverging,
  describe,
  onSelectRow,
  selectedRow,
  colLabels,
  scaleLabels,
  scale = (v) => v,
}: {
  values: number[][];
  min: number;
  max: number;
  diverging: boolean;
  describe: (row: number, col: number, value: number) => ReactNode;
  onSelectRow?: (row: number) => void;
  selectedRow?: number | null;
  colLabels?: string[];
  scaleLabels?: [string, string];
  scale?: (v: number) => number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { show, hide, node } = useTooltip();
  const rows = values.length;
  const cols = values[0]?.length ?? 0;
  const rowH = 14;
  const bound = diverging ? Math.max(Math.abs(min), Math.abs(max), 1e-9) : Math.max(scale(max), 1e-9);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !cols) return;
    const paint = () => {
      const low = cssColor(canvas, "--seq-low");
      const high = cssColor(canvas, "--seq-high");
      const neg = cssColor(canvas, "--div-neg");
      const mid = cssColor(canvas, "--div-mid");
      const ctx = canvas.getContext("2d")!;
      const img = ctx.createImageData(cols, rows);
      values.forEach((row, r) =>
        row.forEach((v, c) => {
          const rgb = diverging
            ? v >= 0
              ? mix(mid, high, Math.min(1, v / bound))
              : mix(mid, neg, Math.min(1, -v / bound))
            : mix(low, high, Math.min(1, Math.max(0, scale(v) / bound)));
          img.data.set([...rgb, 255], (r * cols + c) * 4);
        }),
      );
      ctx.putImageData(img, 0, 0);
    };
    paint();
    // Repaint when the OS theme changes: the ramp ends are theme tokens.
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", paint);
    return () => mq.removeEventListener("change", paint);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values, rows, cols, diverging, bound]);

  const locate = (e: React.MouseEvent) => {
    const box = e.currentTarget.getBoundingClientRect();
    const r = Math.min(rows - 1, Math.max(0, Math.floor(((e.clientY - box.top) / box.height) * rows)));
    const c = Math.min(cols - 1, Math.max(0, Math.floor(((e.clientX - box.left) / box.width) * cols)));
    return [r, c] as const;
  };

  return (
    <div className="chart heatmap">
      {colLabels && (
        <div className="heatmap-cols">
          {colLabels.map((c) => (
            <span key={c}>{c}</span>
          ))}
        </div>
      )}
      <div className="heatmap-body">
        <div className="heatmap-axis" style={{ height: rows * rowH }}>
          {values.map((_, r) => (
            <span key={r} className={selectedRow === r ? "selected" : ""}>
              {r % 2 === 0 || rows <= 16 ? r : ""}
            </span>
          ))}
        </div>
        <canvas
          ref={ref}
          width={cols}
          height={rows}
          style={{ height: rows * rowH, cursor: onSelectRow ? "pointer" : "default" }}
          onMouseMove={(e) => {
            const [r, c] = locate(e);
            show(e, describe(r, c, values[r][c]));
          }}
          onMouseLeave={hide}
          onClick={(e) => onSelectRow?.(locate(e)[0])}
        />
      </div>
      <div className="heatmap-scale">
        <span>{scaleLabels ? scaleLabels[0] : diverging ? `−${(bound * 100).toFixed(0)}%` : "0%"}</span>
        <div className={diverging ? "ramp diverging" : "ramp"} />
        <span>{scaleLabels ? scaleLabels[1] : diverging ? `+${(bound * 100).toFixed(0)}%` : `${(bound * 100).toFixed(0)}%`}</span>
      </div>
      {node}
    </div>
  );
}

export function Tile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {note && <div className="tile-note">{note}</div>}
    </div>
  );
}

export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  label,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.value)} className={o.value === value ? "on" : ""} aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
