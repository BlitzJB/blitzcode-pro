// blitzcode-pro Tauri shell — supervises a Python uvicorn child.
//
// Responsibilities:
//   1. Choose an ephemeral free port (so we never collide with anything).
//   2. Spawn the Python server with that port + the bundled-mode flag +
//      the path to the static frontend.
//   3. Tee the child's stdout/stderr into ~/.agent-webkit/blitzcode-pro/logs/app.log
//      (Python owns server.log via its own rotating file handler; this
//      catches things that bypass the logging system — panics, import
//      errors, hard crashes).
//   4. Wait until the port is accepting connections, then open a WebView
//      pointing at it.
//   5. On window close: SIGTERM the child, wait up to 5s, SIGKILL if
//      still alive.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use tauri::{RunEvent, TitleBarStyle, WebviewUrl, WebviewWindowBuilder};

const READY_TIMEOUT: Duration = Duration::from_secs(30);
const READY_POLL_INTERVAL: Duration = Duration::from_millis(100);
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// JavaScript injected into the WebView before the page loads. Mirrors
/// Tauri's bundled `drag.js` (which only runs for the tauri:// protocol)
/// so any element marked `data-tauri-drag-region` becomes a window-drag
/// handle that also responds to double-click-to-maximize. Bound at
/// mousedown specifically: NSWindow needs the gesture handed off
/// synchronously inside the native event loop.
const DRAG_REGION_SCRIPT: &str = r#"
document.addEventListener('mousedown', function (e) {
    if (e.button !== 0) return;
    var t = e.target;
    if (!(t instanceof Element)) return;
    if (!t.closest('[data-tauri-drag-region]')) return;
    if (t.closest('[data-tauri-drag-region="false"]')) return;
    if (t.closest('button, a, input, textarea, select, [role="button"], [contenteditable="true"]')) return;
    e.preventDefault();
    if (e.detail === 2) {
        window.__TAURI_INTERNALS__.invoke('plugin:window|toggle_maximize');
    } else {
        window.__TAURI_INTERNALS__.invoke('plugin:window|start_dragging');
    }
});
"#;

/// Holds the Python child so we can kill it on shutdown. Wrapped in
/// Mutex<Option<…>> because Tauri's `manage()` requires Send + Sync and
/// we want to be able to `.take()` the child to consume it during exit.
struct Supervisor {
    child: Mutex<Option<Child>>,
}

fn main() {
    if let Err(e) = run() {
        // Last-resort error reporting. Anything before the WebView
        // opens has nowhere else to surface — write to app.log AND
        // stderr so a terminal launch shows it too.
        let _ = log_line(&format!("FATAL: {:#}", e));
        eprintln!("blitzcode shell fatal: {:#}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    rotate_app_log()?;
    log_line("blitzcode shell starting")?;

    let port = pick_free_port().context("choosing an ephemeral port")?;
    log_line(&format!("port chosen: {}", port))?;

    let layout = resolve_layout().context("resolving server + web paths")?;
    log_line(&format!(
        "layout: server={} web={}",
        layout.server_dir.display(),
        layout.web_dist.display()
    ))?;

    let child = spawn_python(&layout, port).context("spawning Python server")?;
    let supervisor = Arc::new(Supervisor {
        child: Mutex::new(Some(child)),
    });

    wait_for_ready(port).context("waiting for server to bind")?;
    log_line("server ready, opening window")?;

    let sup_for_setup = Arc::clone(&supervisor);
    let sup_for_exit = Arc::clone(&supervisor);

    tauri::Builder::default()
        .manage(sup_for_setup)
        .setup(move |app| {
            let url = format!("http://127.0.0.1:{port}/")
                .parse()
                .expect("constructed URL parses");
            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url))
                .title("blitzcode")
                .inner_size(1280.0, 800.0)
                .min_inner_size(960.0, 600.0)
                // Overlay titlebar: removes the system chrome but keeps
                // the traffic lights floating in the top-left over our
                // canvas. `hidden_title` drops the centered text.
                .title_bar_style(TitleBarStyle::Overlay)
                .hidden_title(true)
                // External URLs don't get Tauri's `data-tauri-drag-region`
                // auto-wiring (only the bundled tauri:// protocol does).
                // We inject the same behavior manually. Must be in
                // mousedown (synchronous) so NSWindow can pick up the
                // gesture — a React-side pointerdown listener is too
                // late and silently no-ops.
                .initialization_script(DRAG_REGION_SCRIPT)
                .build()?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .context("building Tauri app")?
        .run(move |_app_handle, event| {
            if let RunEvent::ExitRequested { .. } = event {
                shutdown(&sup_for_exit);
            }
        });

    Ok(())
}

// ─── Layout ─────────────────────────────────────────────────────────────────

struct Layout {
    server_dir: PathBuf,
    web_dist: PathBuf,
}

/// Find apps/server and apps/web/out. In a packaged .app these live in
/// `Contents/Resources/`; in dev (`cargo tauri dev`) they're at the
/// repo root relative to the manifest dir.
fn resolve_layout() -> Result<Layout> {
    // Bundled: resources path from Tauri's PathResolver — but we don't
    // have AppHandle yet here (it's pre-builder). Fall back to walking
    // the executable path.
    if let Ok(exe) = env::current_exe() {
        // .app/Contents/MacOS/blitzcode → .app/Contents/Resources
        if let Some(macos_dir) = exe.parent() {
            if let Some(contents) = macos_dir.parent() {
                let resources = contents.join("Resources");
                let server_dir = resources.join("server");
                let web_dist = resources.join("web");
                if server_dir.is_dir() && web_dist.is_dir() {
                    return Ok(Layout { server_dir, web_dist });
                }
            }
        }
    }
    // Dev: from src-tauri/target/debug/blitzcode-desktop walk up to repo.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // apps/desktop/src-tauri → apps/desktop → apps → repo
    let repo = manifest_dir
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .ok_or_else(|| anyhow!("cannot derive repo root from manifest dir"))?;
    let server_dir = repo.join("apps/server");
    let web_dist = repo.join("apps/web/out");
    if !server_dir.is_dir() {
        return Err(anyhow!("server dir not found at {}", server_dir.display()));
    }
    if !web_dist.is_dir() {
        return Err(anyhow!(
            "web dist not found at {} — run `pnpm build` in apps/web first",
            web_dist.display()
        ));
    }
    Ok(Layout { server_dir, web_dist })
}

// ─── Port + spawn ───────────────────────────────────────────────────────────

/// macOS Finder-launched apps inherit a stripped PATH, so `Command::new("uv")`
/// often fails to find anything. Walk the usual install locations directly.
fn locate_uv() -> Result<PathBuf> {
    let candidates = [
        "/opt/homebrew/bin/uv",
        "/usr/local/bin/uv",
        "/usr/bin/uv",
    ];
    for p in candidates {
        if Path::new(p).exists() {
            return Ok(PathBuf::from(p));
        }
    }
    // ~/.local/bin/uv (astral's curl installer default)
    if let Some(home) = dirs::home_dir() {
        let p = home.join(".local/bin/uv");
        if p.exists() {
            return Ok(p);
        }
        let p = home.join(".cargo/bin/uv");
        if p.exists() {
            return Ok(p);
        }
    }
    // Fall back to PATH lookup — works when launched from a terminal.
    Ok(PathBuf::from("uv"))
}

fn pick_free_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener); // immediate release — kernel keeps the port reserved
                    // briefly under SO_REUSEADDR semantics, but in
                    // practice the next bind() always wins.
    Ok(port)
}

fn spawn_python(layout: &Layout, port: u16) -> Result<Child> {
    // Phase 2: rely on user-side `uv` to resolve + run the Python deps.
    // Phase 3 will swap this for a bundled python-build-standalone
    // interpreter + a pre-installed site-packages dir.
    let uv = locate_uv()
        .context("uv not found — install with `brew install uv` or via https://astral.sh/uv")?;
    let mut cmd = Command::new(&uv);
    cmd.current_dir(&layout.server_dir)
        .arg("run")
        .arg("python")
        .arg("-m")
        .arg("main")
        .env("BLITZCODE_PRO_BUNDLED", "1")
        .env("BLITZCODE_PRO_PORT", port.to_string())
        .env("BLITZCODE_PRO_HOST", "127.0.0.1")
        .env("BLITZCODE_PRO_WEB_DIST", &layout.web_dist)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().with_context(|| format!("spawning {}", uv.display()))?;
    let pid = child.id();

    // Tee both streams into app.log so panics / import errors / anything
    // that bypasses Python's RotatingFileHandler still gets captured.
    if let Some(out) = child.stdout.take() {
        thread::spawn(move || tee("py-stdout", out));
    }
    if let Some(err) = child.stderr.take() {
        thread::spawn(move || tee("py-stderr", err));
    }
    log_line(&format!("spawned python pid={pid} port={port}"))?;
    Ok(child)
}

fn tee<R: std::io::Read>(label: &str, stream: R) {
    let reader = BufReader::new(stream);
    for line in reader.lines().flatten() {
        let _ = log_line(&format!("[{label}] {line}"));
    }
}

fn wait_for_ready(port: u16) -> Result<()> {
    let addr: SocketAddr = format!("127.0.0.1:{port}")
        .parse()
        .expect("valid socket addr");
    let deadline = Instant::now() + READY_TIMEOUT;
    while Instant::now() < deadline {
        if TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok() {
            return Ok(());
        }
        thread::sleep(READY_POLL_INTERVAL);
    }
    Err(anyhow!(
        "server did not bind to port {port} within {:?}",
        READY_TIMEOUT
    ))
}

// ─── Shutdown ───────────────────────────────────────────────────────────────

fn shutdown(supervisor: &Arc<Supervisor>) {
    let mut guard = match supervisor.child.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let Some(mut child) = guard.take() else {
        return;
    };
    let pid = child.id();
    let _ = log_line(&format!("shutting down python pid={pid}"));

    // SIGTERM via Unix signals — std::process::Child only has .kill()
    // (which sends SIGKILL on Unix). For a graceful stop we shell out.
    let term_ok = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()
        .ok()
        .map(|s| s.success())
        .unwrap_or(false);
    if !term_ok {
        let _ = log_line("SIGTERM dispatch failed; falling back to SIGKILL");
        let _ = child.kill();
        let _ = child.wait();
        return;
    }

    // Poll for graceful exit.
    let deadline = Instant::now() + SHUTDOWN_GRACE;
    while Instant::now() < deadline {
        if let Ok(Some(status)) = child.try_wait() {
            let _ = log_line(&format!("python exited cleanly: {status}"));
            return;
        }
        thread::sleep(Duration::from_millis(100));
    }
    let _ = log_line("python did not exit in time; SIGKILL");
    let _ = child.kill();
    let _ = child.wait();
}

// ─── Logging ────────────────────────────────────────────────────────────────

fn log_path() -> PathBuf {
    let base = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".agent-webkit/blitzcode-pro/logs");
    let _ = std::fs::create_dir_all(&base);
    base.join("app.log")
}

fn rotate_app_log() -> Result<()> {
    let path = log_path();
    if let Ok(meta) = std::fs::metadata(&path) {
        // Rotate when the file gets fat. Keep 4 generations.
        if meta.len() > 2_000_000 {
            for i in (1..4).rev() {
                let from = path.with_extension(format!("log.{i}"));
                let to = path.with_extension(format!("log.{}", i + 1));
                let _ = std::fs::rename(from, to);
            }
            let _ = std::fs::rename(&path, path.with_extension("log.1"));
        }
    }
    // Make sure it exists for the first log_line() call.
    let _ = OpenOptions::new().create(true).append(true).open(&path)?;
    Ok(())
}

fn log_line(msg: &str) -> Result<()> {
    let ts = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S");
    let line = format!("{ts}  {msg}\n");
    let mut f: File = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path())?;
    f.write_all(line.as_bytes())?;
    Ok(())
}
