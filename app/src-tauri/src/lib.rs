//! Desktop shell: owns the engine's lifetime and nothing else.
//!
//! On launch it prepares the engine's Python environment (first run only), starts
//! the engine's local API server, and tells the UI where it is. All analysis
//! logic lives in the Python engine; the UI talks to it directly over
//! 127.0.0.1 with a per-launch token. On exit the engine is stopped.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use serde::Serialize;
use tauri::{AppHandle, Manager, RunEvent, State};

/// What the UI polls while the engine comes up.
#[derive(Clone, Serialize)]
struct Status {
    /// "setup" (first-run install), "starting", "ready" or "failed".
    phase: String,
    message: String,
    url: Option<String>,
    token: Option<String>,
}

struct Engine {
    status: Mutex<Status>,
    child: Mutex<Option<Child>>,
}

#[tauri::command]
fn engine_status(engine: State<'_, Engine>) -> Status {
    engine.status.lock().unwrap().clone()
}

/// The UI reports its own problems here, since a desktop window has no visible console.
/// Lines go to the app's standard error, which the launcher or a terminal captures.
#[tauri::command]
fn ui_log(message: String) {
    eprintln!("[ui] {message}");
}

fn set_status(app: &AppHandle, phase: &str, message: impl Into<String>) {
    let engine = app.state::<Engine>();
    let mut status = engine.status.lock().unwrap();
    status.phase = phase.into();
    status.message = message.into();
}

/// Keep child processes from opening a console window on Windows.
fn quiet(command: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    command
}

fn exe_name(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

/// The engine's source: an override, the copy bundled with the app, or the
/// repository checkout when running a development build.
fn engine_dir(app: &AppHandle) -> Result<PathBuf, String> {
    if let Ok(dir) = std::env::var("XRAY_ENGINE_DIR") {
        return Ok(PathBuf::from(dir));
    }
    if let Ok(resources) = app.path().resource_dir() {
        let bundled = resources.join("engine");
        if bundled.join("pyproject.toml").exists() {
            return Ok(bundled);
        }
    }
    let dev = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../engine");
    if dev.join("pyproject.toml").exists() {
        return Ok(dev);
    }
    Err("The engine files are missing from this installation.".into())
}

/// `uv` installs Python and the engine's packages. It ships next to the app's
/// executable; a development build may use one from the usual install places.
fn find_uv() -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var("XRAY_UV") {
        return Ok(PathBuf::from(path));
    }
    let name = exe_name("uv");
    if let Ok(exe) = std::env::current_exe() {
        if let Some(beside) = exe.parent().map(|dir| dir.join(&name)) {
            if beside.exists() {
                return Ok(beside);
            }
        }
    }
    let mut places: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).collect())
        .unwrap_or_default();
    // Apps started from the desktop do not inherit the shell's PATH.
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        let home = PathBuf::from(home);
        places.push(home.join(".local/bin"));
        places.push(home.join(".cargo/bin"));
    }
    places.push(PathBuf::from("/opt/homebrew/bin"));
    places.push(PathBuf::from("/usr/local/bin"));
    places
        .into_iter()
        .map(|dir| dir.join(&name))
        .find(|path| path.exists())
        .ok_or_else(|| "Could not find `uv`, which installs the engine's Python packages.".into())
}

fn last_lines(text: &[u8], count: usize) -> String {
    let text = String::from_utf8_lossy(text);
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(count)..].join("\n")
}

fn start_engine(app: &AppHandle) -> Result<(), String> {
    let data = match std::env::var("XRAY_APP_DATA") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => app.path().app_data_dir().map_err(|e| e.to_string())?,
    };
    std::fs::create_dir_all(&data).map_err(|e| format!("Cannot create {}: {e}", data.display()))?;
    let engine = engine_dir(app)?;
    let venv = data.join("venv");
    let python = if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    };

    // 1. Python environment. `uv sync` is a no-op when it is already up to date,
    //    so this runs on every launch and also applies app updates.
    let first_run = !python.exists();
    set_status(
        app,
        "setup",
        if first_run {
            "Setting up for first use. This downloads Python and PyTorch (about 2 GB) and can take several minutes."
        } else {
            "Checking the Python environment."
        },
    );
    let uv = find_uv()?;
    let sync = quiet(
        Command::new(&uv)
            .args(["sync", "--frozen", "--no-dev", "--project"])
            .arg(&engine)
            .env("UV_PROJECT_ENVIRONMENT", &venv)
            .stdin(Stdio::null()),
    )
    .output()
    .map_err(|e| format!("Could not run uv: {e}"))?;
    if !sync.status.success() {
        return Err(format!(
            "Setting up the Python environment failed:\n{}",
            last_lines(&sync.stderr, 12)
        ));
    }

    // 2. The engine's API server, on a free port with a fresh token.
    set_status(app, "starting", "Starting the engine.");
    let log = std::fs::File::create(data.join("engine.log")).map_err(|e| e.to_string())?;
    let mut log_tail = log.try_clone().map_err(|e| e.to_string())?;
    let mut child = quiet(
        Command::new(&python)
            .args(["-m", "xray.cli", "serve", "--port", "0"])
            .env("XRAY_HOME", data.join("xray"))
            .env("PYTHONUNBUFFERED", "1")
            // The engine stops itself if this process disappears (crash, force quit).
            .env("XRAY_PARENT_PID", std::process::id().to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(log)),
    )
    .spawn()
    .map_err(|e| format!("Could not start the engine: {e}"))?;

    // Its first line of output is {"event": "ready", "port": ..., "token": ...}.
    let mut output = BufReader::new(child.stdout.take().ok_or("The engine has no output")?);
    let mut line = String::new();
    output.read_line(&mut line).map_err(|e| e.to_string())?;
    // Keep reading for the engine's whole life: if nobody reads the pipe, anything
    // the engine prints later fails with a broken pipe.
    std::thread::spawn(move || {
        let _ = std::io::copy(&mut output, &mut log_tail);
    });
    let ready: serde_json::Value = serde_json::from_str(line.trim()).map_err(|_| {
        let _ = child.kill();
        format!(
            "The engine did not start. See {}",
            data.join("engine.log").display()
        )
    })?;
    let port = ready["port"].as_u64().ok_or("The engine reported no port")?;
    let token = ready["token"].as_str().ok_or("The engine reported no token")?;

    let state = app.state::<Engine>();
    *state.child.lock().unwrap() = Some(child);
    let mut status = state.status.lock().unwrap();
    status.phase = "ready".into();
    status.message = "The engine is running.".into();
    status.url = Some(format!("http://127.0.0.1:{port}"));
    status.token = Some(token.to_string());
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(Engine {
            status: Mutex::new(Status {
                phase: "starting".into(),
                message: "Starting.".into(),
                url: None,
                token: None,
            }),
            child: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![engine_status, ui_log])
        .setup(|app| {
            let handle = app.handle().clone();
            // Off the main thread: first-run setup can take minutes and the window
            // must stay responsive to show progress.
            std::thread::spawn(move || {
                if let Err(message) = start_engine(&handle) {
                    set_status(&handle, "failed", message);
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the application");

    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            if let Some(mut child) = handle.state::<Engine>().child.lock().unwrap().take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    });
}
