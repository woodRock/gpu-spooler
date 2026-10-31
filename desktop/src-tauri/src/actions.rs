//! What the window can change: kill, resubmit, clear, submit, start the daemon. Each is the
//! library call the CLI makes, run off the main thread (most of them talk to servers).
//!
//! The daemon is always the installed `task` binary (`start_daemon_with`), never this app:
//! submit and resubmit pass `start: false` and the app starts the daemon afterwards if needed.
//! After each, the watcher reads again at once, so the window shows the result with the toast.

use crate::state::AppState;
use gpu_spooler::config::load_config;
use gpu_spooler::core::{self, Cleared, KillResult, SubmitOpts, Submitted};
use gpu_spooler::paths::Paths;
use gpu_spooler::transport::Transport;
use std::path::{Path, PathBuf};

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, String> + Send + 'static) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(f).await.map_err(|e| e.to_string())?
}

fn transport(paths: &Paths) -> Result<Transport, String> {
    let cfg = load_config(paths).map_err(|e| format!("config: {}", e.0))?;
    Ok(Transport::new(cfg, paths.clone()))
}

/// Whether `exe` is gpu-spooler's `task` (`task 0.2.0`), not another program of that name
/// (Taskwarrior prints a bare version).
fn is_spooler(exe: &Path) -> bool {
    std::process::Command::new(exe)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).starts_with("task "))
}

/// Where to look for `task`. An app started from Finder gets launchd's PATH, which lacks
/// ~/.local/bin (where install.sh puts it), so that comes first.
fn candidates(home: &Path, path_var: Option<&str>) -> Vec<PathBuf> {
    let mut dirs = vec![home.join(".local/bin")];
    dirs.extend(path_var.unwrap_or("").split(':').filter(|d| !d.is_empty()).map(PathBuf::from));
    dirs.extend(["/opt/homebrew/bin", "/usr/local/bin"].map(PathBuf::from));
    dirs.push(home.join(".cargo/bin"));
    let mut out: Vec<PathBuf> = vec![];
    for d in dirs {
        let exe = d.join("task");
        if !out.contains(&exe) {
            out.push(exe);
        }
    }
    out
}

/// The installed `task` binary.
pub fn task_exe(paths: &Paths) -> Result<PathBuf, String> {
    candidates(&paths.home, std::env::var("PATH").ok().as_deref())
        .into_iter()
        .find(|exe| exe.is_file() && is_spooler(exe))
        .ok_or_else(|| "gpu-spooler's `task` was not found in ~/.local/bin or on PATH; install it first".into())
}

/// Start the daemon unless it runs, with the `task` that `exe` finds: its new PID, or None if it
/// was running.
fn ensure_daemon(paths: &Paths, exe: impl FnOnce(&Paths) -> Result<PathBuf, String>) -> Result<Option<i64>, String> {
    if core::daemon_pid(paths).is_some() {
        return Ok(None);
    }
    core::start_daemon_with(paths, &exe(paths)?).map_err(|e| e.to_string())
}

/// After a submit: start the daemon, or say in the warnings why the task won't start yet.
fn started(paths: &Paths, mut sub: Submitted, exe: impl FnOnce(&Paths) -> Result<PathBuf, String>) -> Submitted {
    if let Err(e) = ensure_daemon(paths, exe) {
        sub.warnings.push(format!("Queued, but the daemon could not be started: {e}"));
    }
    sub
}

/// Cancel a queued task or kill a running one.
#[tauri::command]
#[specta::specta]
pub async fn kill_task(state: tauri::State<'_, AppState>, id: i64) -> Result<KillResult, String> {
    let paths = state.paths.clone();
    let r = blocking(move || core::kill(&transport(&paths)?, id).map_err(|e| e.to_string())).await;
    state.refresh();
    r
}

/// Queue a copy of a task (same command, resources and directory).
#[tauri::command]
#[specta::specta]
pub async fn resubmit_task(state: tauri::State<'_, AppState>, id: i64) -> Result<Submitted, String> {
    let paths = state.paths.clone();
    let r = blocking(move || {
        let sub = core::resubmit(&transport(&paths)?, id, false).map_err(|e| e.to_string())?;
        Ok(started(&paths, sub, task_exe))
    })
    .await;
    state.refresh();
    r
}

/// Queue a new task. The repo check (one ssh call per remote copy) runs as `task submit`'s does.
#[tauri::command]
#[specta::specta]
pub async fn submit_task(state: tauri::State<'_, AppState>, opts: SubmitOpts) -> Result<Submitted, String> {
    let paths = state.paths.clone();
    let r = blocking(move || {
        let opts = SubmitOpts { start: false, check_repo: true, ..opts };
        let sub = core::submit(&transport(&paths)?, &opts).map_err(|e| e.to_string())?;
        Ok(started(&paths, sub, task_exe))
    })
    .await;
    state.refresh();
    r
}

/// Remove finished tasks, their logs and their job directories on the servers.
#[tauri::command]
#[specta::specta]
pub async fn clear_finished(state: tauri::State<'_, AppState>) -> Result<Cleared, String> {
    let paths = state.paths.clone();
    let r = blocking(move || core::clear_finished(&transport(&paths)?).map_err(|e| e.to_string())).await;
    state.refresh();
    r
}

/// Start the daemon with the installed `task`: its PID, or None if it was already running.
#[tauri::command]
#[specta::specta]
pub async fn start_daemon(state: tauri::State<'_, AppState>) -> Result<Option<i64>, String> {
    let paths = state.paths.clone();
    let r = blocking(move || ensure_daemon(&paths, task_exe)).await;
    state.refresh();
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn script(dir: &Path, body: &str) -> PathBuf {
        let exe = dir.join("task");
        std::fs::write(&exe, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
        exe
    }

    #[test]
    fn looks_in_local_bin_first_then_path() {
        let c = candidates(Path::new("/h"), Some("/a:/h/.local/bin::/b"));
        let c: Vec<&str> = c.iter().map(|p| p.to_str().unwrap()).collect();
        assert_eq!(
            c,
            [
                "/h/.local/bin/task",
                "/a/task",
                "/b/task",
                "/opt/homebrew/bin/task",
                "/usr/local/bin/task",
                "/h/.cargo/bin/task"
            ]
        );
    }

    #[test]
    fn tells_gpu_spoolers_task_from_taskwarrior() {
        let dir = tempfile::tempdir().unwrap();
        assert!(is_spooler(&script(dir.path(), "echo 'task 0.2.0'")));
        assert!(!is_spooler(&script(dir.path(), "echo '3.4.1'")));
        assert!(!is_spooler(&dir.path().join("missing")));
    }

    fn submitted() -> Submitted {
        Submitted {
            task_id: 1,
            command: "x".into(),
            working_dir: "/".into(),
            num_gpus: 1,
            min_mem: 0,
            name: None,
            host_spec: None,
            candidates: vec![],
            warnings: vec![],
            gpu_type: None,
        }
    }

    #[test]
    fn a_submit_starts_the_daemon_or_says_why_not() {
        let home = tempfile::tempdir().unwrap();
        let paths = Paths::for_home(home.path());
        let w = started(&paths, submitted(), |_| Err("not installed".into())).warnings;
        assert_eq!(w, ["Queued, but the daemon could not be started: not installed"]);

        let bin = tempfile::tempdir().unwrap();
        let exe = script(bin.path(), "exec sleep 30");
        assert!(started(&paths, submitted(), |_| Ok(exe.clone())).warnings.is_empty());
        let pid = core::daemon_pid(&paths).expect("started");
        assert_eq!(ensure_daemon(&paths, |_| panic!("not looked up while it runs")), Ok(None));
        assert_eq!(core::stop_daemon(&paths), Some(pid));
    }
}
