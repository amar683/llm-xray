import { useState } from "react";
import { api, num, useAsync, type AdvisorPreview, type AdvisorResult, type Conn, type Report as ReportData } from "../api";
import { Tile } from "../components/charts";
import type { Model } from "./Models";

const NAMES: Record<string, string> = { timing: "Timing", profile: "Activations", quant: "Quantization", loops: "Loops" };
const STATE: Record<string, string> = { full: "in full", short: "shortened", dropped: "left out", empty: "no data" };

export function Report({ conn, model, onEdit }: { conn: Conn; model: Model; onEdit: (commands: string) => void }) {
  const [budget, setBudget] = useState(3000);
  const report = useAsync(() => api<ReportData>(conn, "/report", { body: { model: model.path, budget } }), [conn, model.path, budget]);
  const status = useAsync(() => api<{ sdk_installed: boolean; credential_in_environment: boolean }>(conn, "/advisor/status"), [conn]);

  const [question, setQuestion] = useState("layer 12");
  const [answers, setAnswers] = useState<{ query: string; answer: string; failed?: boolean }[]>([]);
  const ask = async () => {
    const query = question.trim();
    if (!query) return;
    try {
      const r = await api<{ answer: string }>(conn, "/query", { body: { model: model.path, query } });
      setAnswers((a) => [{ query, answer: r.answer }, ...a]);
    } catch (e) {
      setAnswers((a) => [{ query, answer: (e as Error).message, failed: true }, ...a]);
    }
  };

  const [goal, setGoal] = useState("");
  const [preview, setPreview] = useState<AdvisorPreview | null>(null);
  const [result, setResult] = useState<AdvisorResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const body = { model: model.path, goal, budget };
  const loadPreview = async () => {
    setError(null);
    setResult(null);
    try {
      setPreview(await api<AdvisorPreview>(conn, "/advisor/preview", { body }));
    } catch (e) {
      setPreview(null);
      setError((e as Error).message);
    }
  };
  const send = async () => {
    setError(null);
    setBusy(true);
    try {
      setResult(await api<AdvisorResult>(conn, "/advisor/send", { body }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const r = report.data;
  const ready = status.data?.sdk_installed && status.data?.credential_in_environment;

  return (
    <>
      <h1>Report</h1>
      <p className="summary">Everything measured for {model.id}, condensed to fit a token budget. Built from saved runs on this machine.</p>

      {report.error && <p className="error">{report.error}</p>}
      {r && (
        <>
          <div className="tiles">
            <Tile label="Report size" value={`${num(r.tokens_estimate)} tokens`} note={r.fits ? `within the ${num(r.budget_tokens)} budget` : `over the ${num(r.budget_tokens)} budget`} />
            <Tile label="Sections cut" value={String(r.sections.filter((s) => s.state === "short" || s.state === "dropped").length)} note="shortened or left out to fit" />
            {r.missing.length > 0 && <Tile label="No run yet for" value={r.missing.map((m) => NAMES[m] ?? m).join(", ")} />}
          </div>

          <section className="panel">
            <div className="panel-head">
              <h2>Report</h2>
              <label className="inline">
                Token budget
                <input type="number" min={300} max={20000} step={100} value={budget} onChange={(e) => setBudget(Number(e.target.value) || 300)} />
              </label>
            </div>
            <pre className="report">{r.text}</pre>
            <p className="muted">
              {r.sections.map((s) => `${s.name}: ${STATE[s.state] ?? s.state}`).join(" · ")}. Token count is a cautious estimate (2.2 characters per token). The text contains no file paths or
              prompts.
            </p>
          </section>
        </>
      )}

      <section className="panel">
        <h2>Ask for a detail</h2>
        <form
          className="row"
          onSubmit={(e) => {
            e.preventDefault();
            ask();
          }}
        >
          <input aria-label="Query" value={question} onChange={(e) => setQuestion(e.target.value)} />
          <button type="submit">Ask</button>
        </form>
        <p className="muted">
          Forms: <code>layer N</code>, <code>neuron L N</code>, <code>top_neurons L SET</code>, <code>head L H</code>, <code>tensor L KIND</code>. Answered from saved runs; nothing is sent
          anywhere.
        </p>
        {answers.map((a, i) => (
          <p key={i} className={a.failed ? "error" : ""}>
            <b>{a.query}</b>: {a.answer}
          </p>
        ))}
      </section>

      <section className="panel">
        <h2>Optional: ask an AI advisor</h2>
        <p>
          This sends the report above and your goal to the Claude API and asks what edit to try next. It is off unless you use it here. No weights, prompts or file paths are sent. If the
          advisor asks for details, those short answers are sent too.
        </p>
        <label>
          Your goal
          <input value={goal} onChange={(e) => setGoal(e.target.value)} placeholder="e.g. Reduce looping without losing capitals accuracy" />
        </label>
        <div className="run-actions" style={{ marginTop: 12 }}>
          <button onClick={loadPreview} disabled={!goal.trim()}>
            Preview what would be sent
          </button>
          <button className="primary" onClick={send} disabled={!preview || !ready || busy}>
            {busy ? "Waiting for the advisor…" : "Send to the Claude API"}
          </button>
          {status.data && !ready && (
            <span className="muted">
              {status.data.sdk_installed
                ? "Sending is unavailable: no ANTHROPIC_API_KEY is set where the engine runs."
                : "Sending is unavailable: the engine was installed without the advisor extra."}
            </span>
          )}
        </div>
        {error && <p className="error">{error}</p>}

        {preview && !result && (
          <>
            <h3>
              Exactly what would be sent (about {num(preview.tokens_estimate)} tokens, to {preview.payload.model})
            </h3>
            <details>
              <summary>Instructions to the advisor</summary>
              <pre className="report">{preview.payload.system}</pre>
              <ul className="plain">
                {preview.payload.tools.map((t) => (
                  <li key={t.name}>
                    <code>{t.name}</code>: {t.description}
                  </li>
                ))}
              </ul>
            </details>
            <pre className="report">{preview.payload.messages[0].content}</pre>
          </>
        )}

        {result && (
          <>
            <h3>{result.commands.length ? "Proposed edit" : "The advisor recommends no edit"}</h3>
            {result.queries.length > 0 && <p className="muted">It first asked: {result.queries.map((q) => q.query).join("; ")}.</p>}
            {result.commands.length > 0 ? (
              <>
                <ul className="plain mono">
                  {result.commands_as_proposed.map((c, i) => (
                    <li key={i}>{c}</li>
                  ))}
                </ul>
                <p>
                  <b>Why:</b> {result.rationale}
                </p>
                <p>
                  <b>Expected:</b> {result.expected_effect}
                </p>
                <p>
                  <button className="primary" onClick={() => onEdit(result.commands.join("\n"))}>
                    Open in Edits to measure it
                  </button>{" "}
                  <span className="muted">Nothing is applied until you run it there.</span>
                </p>
              </>
            ) : (
              <p>{result.message}</p>
            )}
            <p className="muted">
              {result.rounds} round{result.rounds === 1 ? "" : "s"}, {num(result.usage.input_tokens)} tokens in, {num(result.usage.output_tokens)} out.
            </p>
          </>
        )}
      </section>
    </>
  );
}
