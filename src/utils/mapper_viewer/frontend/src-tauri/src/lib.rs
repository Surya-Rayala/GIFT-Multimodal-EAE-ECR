// Tauri shell for the Mapper Viewer.
//
// Lifecycle:
//   1. On startup, spawn the Python sidecar
//      (`python -m src.utils.mapper_viewer.backend.sidecar_main --port 0`)
//      from the project root, capture its stdout, and parse the
//      `PORT=<n>` line that the sidecar prints once it has bound a port.
//   2. Store the port in shared state and:
//        a) inject `window.__SIDECAR_PORT__` via eval (early, best effort), and
//        b) expose a `get_sidecar_port` Tauri command the frontend can invoke.
//   3. On window close / app exit, kill the sidecar process.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use tauri::{Manager, RunEvent};

#[derive(Default)]
struct SidecarState {
    process: Mutex<Option<Child>>,
    port: Mutex<Option<u16>>,
}

#[tauri::command]
fn get_sidecar_port(state: tauri::State<'_, SidecarState>) -> Option<u16> {
    *state.port.lock().expect("sidecar port mutex poisoned")
}

/// Best-effort project-root resolver for dev mode. The Cargo manifest sits
/// at `<project>/src/utils/mapper_viewer/frontend/src-tauri/Cargo.toml`, so
/// going five ancestors up reaches the project root.
fn project_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .nth(5)
        .map(|p| p.to_path_buf())
        .unwrap_or(manifest)
}

fn python_executable() -> &'static str {
    option_env!("MAPPER_VIEWER_PYTHON").unwrap_or("python")
}

fn spawn_sidecar() -> Result<(Child, u16), String> {
    let cwd = project_root();
    let python = python_executable();

    let mut child = Command::new(python)
        .args([
            "-m",
            "src.utils.mapper_viewer.backend.sidecar_main",
            "--port",
            "0",
            "--log-level",
            "warning",
        ])
        .current_dir(&cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| {
            format!(
                "Failed to start Python sidecar from {cwd:?} using `{python}`: {e}. \
                 Make sure your Python environment is active and the project's \
                 dependencies (fastapi, uvicorn, opencv-python) are installed."
            )
        })?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Failed to capture sidecar stdout".to_string())?;

    let mut port: Option<u16> = None;
    let reader = BufReader::new(stdout);
    for line in reader.lines().take(50) {
        let line = line.map_err(|e| format!("Sidecar stdout error: {e}"))?;
        if let Some(rest) = line.strip_prefix("PORT=") {
            if let Ok(p) = rest.trim().parse::<u16>() {
                port = Some(p);
                break;
            }
        }
    }

    let port = port.ok_or_else(|| {
        "Sidecar exited or failed to print PORT=<n> within 50 lines.".to_string()
    })?;
    Ok((child, port))
}

fn kill_sidecar(state: &SidecarState) {
    if let Ok(mut guard) = state.process.lock() {
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let (child, port) = match spawn_sidecar() {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("[mapper-viewer] sidecar startup failed: {e}");
            std::process::exit(1);
        }
    };

    let state = SidecarState::default();
    *state.process.lock().unwrap() = Some(child);
    *state.port.lock().unwrap() = Some(port);

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .manage(state)
        .invoke_handler(tauri::generate_handler![get_sidecar_port])
        .setup(move |app| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.eval(&format!(
                    "window.__SIDECAR_PORT__ = {port};"
                ));
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app, event| {
        if let RunEvent::ExitRequested { .. } = event {
            if let Some(state) = app.try_state::<SidecarState>() {
                kill_sidecar(&state);
            }
        }
    });
}
