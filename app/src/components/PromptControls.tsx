// Prompt entry shared by the per-prompt views.
export function PromptControls({
  prompt,
  setPrompt,
  raw,
  setRaw,
}: {
  prompt: string;
  setPrompt: (v: string) => void;
  raw: boolean;
  setRaw: (v: boolean) => void;
}) {
  return (
    <>
      <label className="grow">
        Prompt
        <input value={prompt} onChange={(e) => setPrompt(e.target.value)} />
      </label>
      <label className="check" title="Feed the text as it is. Use this for base models that were not trained on chat.">
        <input type="checkbox" checked={raw} onChange={(e) => setRaw(e.target.checked)} />
        Plain text (no chat template)
      </label>
    </>
  );
}

export const show = (token: string) => token.replace(/\n/g, "↵").replace(/ /g, "·") || "∅";
