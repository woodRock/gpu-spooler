//! Watches ~/.gpu-spooler and pushes a fresh [`QueueState`] to the window when the queue changes.
//!
//! One SQLite connection stays open for the app's lifetime. Opening and closing one per read would
//! checkpoint the WAL whenever the daemon is stopped, and that is itself a change to the folder
//! being watched.

use crate::queue::{self, QueueState};
use gpu_spooler::db;
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

/// The latest state, for the window's first read (`queue_state`) before any event arrives.
#[derive(Default)]
pub struct Latest(pub Mutex<Option<QueueState>>);

/// The files whose changes can change the queue view.
fn relevant(path: &Path) -> bool {
    matches!(path.file_name().and_then(|n| n.to_str()), Some("tasks.db" | "tasks.db-wal" | "daemon.pid" | "hosts.toml"))
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

pub fn spawn(app: AppHandle, paths: Paths) {
    std::thread::Builder::new()
        .name("queue-watch".into())
        .spawn(move || run(&app, &paths))
        .expect("could not start the queue watcher");
}

fn run(app: &AppHandle, paths: &Paths) {
    let (tx, rx) = mpsc::channel::<()>();
    // Kept alive for the loop; without it (no folder yet, or no file events) the poll still runs.
    let _watcher = std::fs::create_dir_all(&paths.state).ok().and_then(|_| {
        let mut w = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            if res.is_ok_and(|e| e.paths.iter().any(|p| relevant(p))) {
                let _ = tx.send(());
            }
        })
        .ok()?;
        w.watch(&paths.state, RecursiveMode::NonRecursive).ok()?;
        Some(w)
    });

    let latest = app.state::<Latest>();
    let mut con = None;
    loop {
        if con.is_none() {
            con = db::open(paths).ok();
        }
        let state = match &con {
            Some(c) => queue::read(c, paths),
            None => QueueState {
                tasks: vec![],
                daemon_pid: gpu_spooler::core::daemon_pid(paths),
                gpus_in_use: 0,
                max_gpus: None,
                error: Some(format!("database: could not open {}", paths.db().display())),
            },
        };
        let mut last = latest.0.lock().unwrap();
        if let Some(state) = changed(&mut last, state) {
            let _ = state.emit(app);
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
        assert!(!relevant(Path::new("/h/.gpu-spooler/gpu_snapshot.json")));
    }
}
