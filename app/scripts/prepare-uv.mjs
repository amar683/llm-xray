// Copy the `uv` on this machine next to the desktop shell's sources, named for
// the build target, so Tauri bundles it with the app (as `uv` beside the
// executable). The app uses it to install Python and the engine's packages on
// first run, so users need nothing preinstalled.
import { execFileSync } from "node:child_process";
import { chmodSync, copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const windows = process.platform === "win32";
const run = (cmd, args) => execFileSync(cmd, args, { encoding: "utf8" }).trim();

const host = /host: (\S+)/.exec(run("rustc", ["-vV"]))?.[1];
if (!host) throw new Error("Could not read the build target from `rustc -vV`.");
const target = process.env.TAURI_ENV_TARGET_TRIPLE || host;

const found = run(windows ? "where" : "which", ["uv"]).split(/\r?\n/)[0];
if (!found || !existsSync(found)) throw new Error("`uv` is not installed (https://docs.astral.sh/uv/).");

const out = join(here, "..", "src-tauri", "binaries", `uv-${target}${windows ? ".exe" : ""}`);
mkdirSync(dirname(out), { recursive: true });
copyFileSync(found, out);
if (!windows) chmodSync(out, 0o755);
console.log(`uv ${run(found, ["--version"])} -> ${out}`);
