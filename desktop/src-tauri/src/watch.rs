//! Watches ~/.gpu-spooler and pushes a fresh [`QueueState`] and [`GpuState`] to the window when
//! the queue or the GPU snapshot changes.

use crate::gpus::{self, GpuState};
use crate::queue::{self, QueueState};
use crate::state::AppState;
use gpu_spooler::paths::Paths;
use notify::{RecursiveMode, Watcher};
use std::path::Path;
use std::sync::{mpsc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Manager};
use tauri_specta::Event;

/// Writes come in bursts (the daemon updates several rows at once): read once they settle.
pub const DEBOUNCE: Duration = Duration::from_millis(250);
/// Read this often regardless: file events can be coalesced or missed for WAL writes.
pub const POLL: Duration = Duration::from_secs(5);

/// The latest states, for the window's first read (`queue_state`, `gpu_state`) before any event.
#[derive(Default)]
pub struct Latest {
    pub queue: Mutex<Option<QueueState>>,
    pub gpus: Mutex<Option<GpuState>>,
}

/// The files whose changes can change the queue view.
fn relevant(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|n| n.to_str()),
        Some("tasks.db" | "tasks.db-wal" | "daemon.pid" | "hosts.toml" | "gpu_snapshot.json")
    )
}

/// The state to send, if it differs from the last one sent. A failed read keeps the last tasks,
/// so a locked or missing database shows an error over the queue instead of an empty table.
pub fn changed(last: &mut Option<QueueState>, mut state: QueueState) -> Option<QueueState> {
    if let (Some(prev), Some(_)) = (last.as_ref(), state.error.as_ref()) {
        if state.tasks.is_empty() {
            state.tasks = prev.tasks.clone();
            state.gpus_in_use = prev.gpus_in_use;
        }
    }
    if last.as_ref() == Some(&state) {
        return None;
    }
    *last = Some(state.clone());
    Some(state)
}

pub fn spawn(app: AppHandle) {
    std::thread::Builder::new().name("watch".into()).spawn(move || run(&app)).expect("could not start the watcher");
}

/// One read of everything the window shows.
fn read(state: &AppState) -> (QueueState, GpuState) {
    let paths = &state.paths;
    let cfg = gpu_spooler::config::load_config(paths).map_err(|e| e.0);
    let queue = state.with_con(|c| queue::read(c, paths, cfg.as_ref().map_err(String::as_str))).unwrap_or_else(|e| {
        QueueState {
            tasks: vec![],
            daemon_pid: gpu_spooler::core::daemon_pid(paths),
            gpus_in_use: 0,
            max_gpus: None,
            error: Some(format!("database: {e}")),
        }
    });
    let running: Vec<_> = queue.tasks.iter().filter(|t| t.status == "running").cloned().collect();
    let snapshot = gpu_spooler::core::read_gpu_snapshot(paths);
    let gpus = gpus::build(snapshot.as_ref(), cfg.as_ref().ok(), &running, &gpus::idle(cfg.as_ref().ok()));
    (queue, gpus)
}

fn watch(paths: &Paths, tx: mpsc::Sender<()>) -> Option<notify::RecommendedWatcher> {
    std::fs::create_dir_all(&paths.state).ok()?;
    let mut w = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if res.is_ok_and(|e| e.paths.iter().any(|p| relevant(p))) {
            let _ = tx.send(());
        }
    })
    .ok()?;
    w.watch(&paths.state, RecursiveMode::NonRecursive).ok()?;
    Some(w)
}

fn run(app: &AppHandle) {
    let state = app.state::<AppState>();
    let latest = app.state::<Latest>();
    let (tx, rx) = mpsc::channel::<()>();
    // Kept alive for the loop; without it (no folder yet, or no file events) the poll still runs.
    let _watcher = watch(&state.paths, tx);
    loop {
        let (queue, gpus) = read(&state);
        if let Some(queue) = changed(&mut latest.queue.lock().unwrap(), queue) {
            let _ = queue.emit(app);
        }
        let mut last = latest.gpus.lock().unwrap();
        if last.as_ref() != Some(&gpus) {
            *last = Some(gpus.clone());
            let _ = gpus.emit(app);
        }
        drop(last);

        match rx.recv_timeout(POLL) {
            Ok(()) => {
                std::thread::sleep(DEBOUNCE);
                while rx.try_recv().is_ok() {}
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            // No watcher: poll only.
            Err(mpsc::RecvTimeoutError::Disconnected) => std::thread::sleep(POLL),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_spooler::db::Task;

    fn state(ids: &[i64], error: Option<&str>) -> QueueState {
        QueueState {
            tasks: ids.iter().map(|&id| Task { id, ..Default::default() }).collect(),
            daemon_pid: None,
            gpus_in_use: ids.len() as i64,
            max_gpus: None,
            error: error.map(String::from),
        }
    }

    #[test]
    fn sends_only_changes() {
        let mut last = None;
        assert!(changed(&mut last, state(&[1], None)).is_some());
        assert!(changed(&mut last, state(&[1], None)).is_none());
        assert_eq!(changed(&mut last, state(&[2, 1], None)).unwrap().tasks.len(), 2);
    }

    #[test]
    fn a_failed_read_keeps_the_last_tasks() {
        let mut last = None;
        changed(&mut last, state(&[2, 1], None));
        let sent = changed(&mut last, state(&[], Some("database: locked"))).unwrap();
        assert_eq!((sent.tasks.len(), sent.gpus_in_use, sent.error.as_deref()), (2, 2, Some("database: locked")));
        assert!(changed(&mut last, state(&[], Some("database: locked"))).is_none());
    }

    #[test]
    fn watches_the_queue_files_only() {
        assert!(relevant(Path::new("/h/.gpu-spooler/tasks.db-wal")));
        assert!(relevant(Path::new("/h/.gpu-spooler/daemon.pid")));
        assert!(!relevant(Path::new("/h/.gpu-spooler/tasks.db-shm")));
        assert!(relevant(Path::new("/h/.gpu-spooler/gpu_snapshot.json")));
        assert!(!relevant(Path::new("/h/.gpu-spooler/gpu_snapshot.request")));
    }
}
