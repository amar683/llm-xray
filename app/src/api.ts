// Client for the local engine API.
import { useCallback, useEffect, useRef, useState } from "react";

export type Conn = { url: string; token: string };

const KEY = "xray.engine";

/** Engine address: build-time env (dev), then whatever the user last connected to. */
export function loadConn(): Conn | null {
  const url = import.meta.env.VITE_ENGINE_URL as string | undefined;
  const token = import.meta.env.VITE_ENGINE_TOKEN as string | undefined;
  if (url && token) return { url, token };
  try {
    const saved = sessionStorage.getItem(KEY);
    return saved ? (JSON.parse(saved) as Conn) : null;
  } catch {
    return null;
  }
}

export function saveConn(conn: Conn | null) {
  try {
    if (conn) sessionStorage.setItem(KEY, JSON.stringify(conn));
    else sessionStorage.removeItem(KEY);
  } catch {
    /* storage unavailable: the connection just won't be remembered */
  }
}

/** True inside the desktop app, where the shell starts the engine itself. */
export const inDesktopApp = () => typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

export type EngineStatus = { phase: "setup" | "starting" | "ready" | "failed"; message: string; url: string | null; token: string | null };

/** Ask the desktop shell how the engine is doing (first-run setup, starting, ready, failed). */
export async function engineStatus(): Promise<EngineStatus> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<EngineStatus>("engine_status");
}

export async function api<T>(conn: Conn, path: string, init?: { method?: string; body?: unknown }): Promise<T> {
  const res = await fetch(conn.url + path, {
    method: init?.method ?? (init?.body ? "POST" : "GET"),
    headers: { Authorization: `Bearer ${conn.token}`, "Content-Type": "application/json" },
    body: init?.body ? JSON.stringify(init.body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [state, setState] = useState<{ data?: T; error?: string; loading: boolean }>({ loading: true });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let live = true;
    setState((s) => ({ ...s, loading: true, error: undefined }));
    fn().then(
      (data) => live && setState({ data, loading: false }),
      (e: Error) => live && setState({ error: e.message, loading: false }),
    );
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { ...state, reload };
}

/** In the desktop app, pass a line to the shell's log. A no-op in a browser. */
export function uiLog(message: string) {
  if (!inDesktopApp()) return;
  import("@tauri-apps/api/core").then(({ invoke }) => invoke("ui_log", { message })).catch(() => undefined);
}

// -- types (only the fields the UI reads) -----------------------------------------

export type ModelInfo = {
  id: string;
  path: string;
  model_type: string | null;
  n_layers: number | null;
  size_bytes: number;
  supported: boolean;
};

export type JobEvent = {
  run_id: string;
  type: string;
  event: "queued" | "started" | "progress" | "done" | "failed" | "cancelled";
  stage?: string;
  done?: number | null;
  total?: number | null;
  error?: string;
};

export type RunRow = {
  id: string;
  type: string;
  status: string;
  backend: string | null;
  started_at: string;
  error: string | null;
  model_path: string | null;
  params: Record<string, unknown>;
};

export type Anatomy = {
  path: string;
  architecture: Record<string, number | string | boolean | null>;
  totals: { params: number; bytes_on_disk: number; stored_dtypes: Record<string, number> };
  by_component: Record<string, number>;
  layers: { index: number; params: number }[];
  memory_estimates: { weights_bytes: Record<string, number>; kv_cache_bytes_fp16: Record<string, number> };
};

export type FitRow = { option: string; need_bytes: number; verdict: string };
export type Fit = {
  device: string;
  context: number;
  available_bytes: number;
  usable_bytes: number;
  analysis: FitRow[];
  running: FitRow[];
  recommended: { analysis: string | null; running: string | null };
};

export type TimingLayer = { layer: number; est_ms: number; attn_ms: number; mlp_ms: number; other_ms: number; share: number };
export type TimingPhase = {
  wall_ms: number;
  tokens_per_s: number;
  dispatch_fraction: number;
  shares: Record<string, number>;
  layers: TimingLayer[];
  other: Record<string, { est_ms: number; share: number }>;
};
export type TimingResult = {
  contexts: { context: number; prefill: TimingPhase; decode: TimingPhase }[];
  bandwidth: {
    peak_gb_per_s: number;
    weights_read_per_token_gb: number;
    decode_floor_ms: number;
    decode_actual_ms: number;
    times_above_floor: number;
  };
  layer_devices: { main: string; offloaded_layers: number[] };
};

export type ProfileLayer = {
  layer: number;
  mean_firing_rate: number;
  hot_fraction: number;
  cold_fraction: number;
  in_out_cosine: number;
  residual_norm: number;
};
export type ProfileResult = {
  sets: Record<string, { prompts: number; tokens: number; layers: ProfileLayer[] }>;
  all: { tokens: number; layers: ProfileLayer[] };
  comparison: Record<
    string,
    {
      specific_fraction: number;
      unused_fraction: number;
      top_specific: { layer: number; neuron: number; rate: number; rate_elsewhere: number }[];
    }
  >;
  redundancy: {
    prompts: number;
    positions: number;
    most_removable: number[];
    layers: { layer: number; drop_kl: number; drop_top1_agreement: number; drop_nll_increase: number; in_out_cosine: number }[];
  } | null;
  run: { prompts: number; generated_tokens: number; seconds: number };
  memory: { peak_gb: number; growth_gb: number };
};

export type QuantTensor = {
  name: string;
  layer: number | null;
  component: string;
  kind: string;
  params: number;
  outlier_ratio: number;
  weighted_error: Record<string, number>;
  kl: number;
  top1_agreement: number;
  kl_by_bits: Record<string, number>;
};
export type QuantVariant = { bytes: number; avg_bpw: number; kl: number; top1_agreement: number; nll_increase: number };
export type QuantPlan = {
  target_bits: number;
  uniform: QuantVariant & { bits: number };
  mixed: QuantVariant & { assignment: Record<string, number>; predicted_kl: number; params_by_bits: Record<string, number> };
};
export type QuantResult = {
  settings: { probe_bits: number; bits_options: number[]; calibration_prompts: number; heldout_prompts: number; block: number };
  full_precision_bytes: number;
  tensors: QuantTensor[];
  layers: { layer: number; kl: number }[];
  plans: QuantPlan[];
};

export type CostValue = {
  layers: {
    layer: number;
    params: number;
    bytes: number;
    decode_ms?: number;
    decode_share?: number;
    skip_kl?: number;
    skip_top1?: number;
    in_out_cosine?: number;
    quant_kl?: number;
  }[];
  sources: Record<string, { id: string; started_at: string }>;
  missing: string[];
  quant_probe_bits: number | null;
};

export type LensResult = {
  prompt: string;
  generated_text: string;
  input_tokens: string[];
  predicted_tokens: string[];
  prob: number[][];
  top: string[][];
  answer_layer: number[];
  mean_answer_layer: number;
};

export type AttributionResult = {
  prompt: string;
  target: string;
  runner_up: string;
  logit_diff: number;
  direct_total: number;
  components: { layer: number; attn_direct: number; mlp_direct: number; attn_ablate: number; mlp_ablate: number }[];
  head_direct: number[][];
  heads: { layer: number; head: number; direct: number; ablate: number }[];
  neurons: { layer: number; neuron: number; attribution: number; activation: number }[];
};

export type LoopSummary = { prompts: number; looped: number; loop_rate: number; ended_normally: number; hit_limit_without_loop: number };
export type LoopsResult = {
  new_tokens: number;
  summary: LoopSummary;
  by_set: Record<string, LoopSummary>;
  cases: { index: number; set: string; prompt: string; n_tokens: number; ended: boolean; loop: { period: number; onset: number } | null; tail: string }[];
  analysis: {
    prompt: string;
    set: string;
    period: number;
    onset: number;
    loop_text: string;
    tokens: string[];
    columns: number[];
    recurrence: number[][];
    recurrence_in_loop: number[];
    recurrence_before: number[];
    locked_from_layer: number | null;
    copy_attention: number[][];
    copy_heads: { layer: number; head: number; attention: number }[];
    writers: { layer: number; kind: string; in_loop: number; before: number }[];
    direction: { layer: number; separation_sigma: number; warning_tokens: number | null; file?: string } | null;
  } | null;
  causal: {
    looping_prompts_tested: number;
    healthy_prompts_tested: number;
    tests: { intervention: string; still_looping: number; now_end_normally: number; healthy_tokens_changed: number; healthy_now_loop: number; healthy_still_end: number }[];
  } | null;
};

export type EditStage = {
  tasks: Record<string, { n: number; correct: number; accuracy: number; stderr: number }>;
  drift: { kl: number; top1_agreement: number; nll_increase: number };
  health: { prompts: number; looped: number; ended: number };
  size: { params: number; fp16_bytes: number; layers: number; mlp_width: number };
};
export type EditResult = {
  commands: string[];
  applied: string[];
  stages: Record<string, EditStage>;
  healing: { steps: number; rank: number; trained_params: number; train_prompts: number; loss_start: number; loss_end: number } | null;
  saved: { path: string; weights_bytes: number } | null;
  tasks: string[];
  code_isolation: string | null;
  patch_id: string | null;
};

export type Patch = {
  id: string;
  name: string;
  parent_id: string | null;
  commands: string[];
  effective_commands: string[];
  chain: { id: string; name: string }[];
  created_at: string;
};

export type CompareRow = { label: string; format: string; a: number | null; b: number | null; change: number | null; verdict: string | null };
export type Comparison = { a: string; b: string; type: string; rows: CompareRow[] };

export type Report = {
  model: string;
  text: string;
  tokens_estimate: number;
  budget_tokens: number;
  fits: boolean;
  sections: { name: string; state: string; tokens: number }[];
  missing: string[];
};

export type AdvisorPreview = {
  payload: { model: string; effort: string; system: string; tools: { name: string; description: string }[]; messages: { role: string; content: string }[] };
  tokens_estimate: number;
  report_tokens: number;
};

export type AdvisorResult = {
  commands: string[];
  commands_as_proposed: string[];
  rationale: string;
  expected_effect: string;
  message: string;
  queries: { query: string; answer: string }[];
  usage: { input_tokens: number; output_tokens: number };
  rounds: number;
};

export type Runtimes = {
  rows: { runtime: string; variant: string; file_bytes: number | null; prefill_tps: number; decode_tps: number; vs_pytorch: number | null }[];
  bench: { id: string; started_at: string; n_prompt: number; n_gen: number; skipped: { runtime: string; reason: string }[] } | null;
  has_timing: boolean;
  llama_cpp_configured: boolean;
};

export type RunDetail<R> = { run: RunRow & { manifest: { backend: Record<string, string> } | null }; result: R | null };

export type Heatmap = { set: string; mode: string; layers: number; neurons: number; bins: number; min: number; max: number; values: number[][] };

// -- live job events ----------------------------------------------------------------

/** Latest event per run, kept current over the engine's WebSocket. */
export function useJobEvents(conn: Conn | null) {
  const [events, setEvents] = useState<Record<string, JobEvent>>({});
  const [online, setOnline] = useState(false);
  const retry = useRef(0);

  useEffect(() => {
    if (!conn) return;
    let ws: WebSocket | null = null;
    let timer: number | undefined;
    let closed = false;

    const connect = () => {
      ws = new WebSocket(`${conn.url.replace(/^http/, "ws")}/events?token=${encodeURIComponent(conn.token)}`);
      ws.onopen = () => {
        retry.current = 0;
        setOnline(true);
      };
      ws.onmessage = (m) => {
        const e = JSON.parse(m.data) as JobEvent;
        setEvents((prev) => ({ ...prev, [e.run_id]: e }));
      };
      ws.onclose = () => {
        setOnline(false);
        if (!closed) timer = window.setTimeout(connect, Math.min(5000, 500 * 2 ** retry.current++));
      };
    };
    connect();
    return () => {
      closed = true;
      window.clearTimeout(timer);
      ws?.close();
    };
  }, [conn]);

  return { events, online };
}

export const isActive = (e?: JobEvent) => !!e && ["queued", "started", "progress"].includes(e.event);

// -- formatting ---------------------------------------------------------------------

export const gb = (bytes: number) => `${(bytes / 2 ** 30).toFixed(2)} GB`;
export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
export const num = (x: number) => x.toLocaleString("en-US");
export const modelName = (path: string | null) => {
  if (!path) return "";
  const m = path.match(/models--([^/]+)--([^/]+)/);
  return m ? `${m[1]}/${m[2]}` : path.split("/").filter(Boolean).pop() ?? path;
};
