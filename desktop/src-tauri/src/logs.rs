//! One task's details and log, for the detail pane. The window follows a running task's log by
//! asking for what was appended since its offset (the daemon copies new output every 5 s).

use crate::state::AppState;
use gpu_spooler::db::Task;
use gpu_spooler::paths::Paths;
use serde::Serialize;
use std::path::PathBuf;

/// How much of a log the pane loads at first; older output stays in the file.
pub const TAIL_BYTES: u64 = 256 * 1024;

#[derive(Clone, Debug, PartialEq, Serialize, specta::Type)]
pub struct LogChunk {
    pub text: String,
    /// Where the next `log_since` continues from.
    pub offset: u64,
    /// The text starts partway through the log (it is longer than TAIL_BYTES).
    pub truncated: bool,
    /// The text replaces what the window has (the log was cleared or rewritten), rather than
    /// following it.
    pub reset: bool,
    /// No log yet: the task hasn't started, or `task clear` removed it.
    pub missing: bool,
    pub path: String,
}

/// Where a task's log is: as recorded, or `logs/<id>.log`.
pub fn log_path(t: &Task, paths: &Paths) -> PathBuf {
    match t.log_file.as_ref().filter(|l| !l.is_empty()) {
        Some(l) => PathBuf::from(l),
        None => paths.logs().join(format!("{}.log", t.id)),
    }
}

fn missing(path: &std::path::Path) -> LogChunk {
    LogChunk {
        text: String::new(),
        offset: 0,
        truncated: false,
        reset: true,
        missing: true,
        path: path.display().to_string(),
    }
}

/// The last TAIL_BYTES of the log.
pub fn tail(path: &std::path::Path) -> LogChunk {
    match gpu_spooler::core::read_log_tail(path, TAIL_BYTES) {
        Ok((text, size)) => LogChunk {
            text,
            offset: size,
            truncated: size > TAIL_BYTES,
            reset: true,
            missing: false,
            path: path.display().to_string(),
        },
        Err(_) => missing(path),
    }
}

/// What was appended since `offset`; the tail again if the file is now shorter than that.
pub fn since(path: &std::path::Path, offset: u64) -> LogChunk {
    match std::fs::metadata(path) {
        Err(_) => missing(path),
        Ok(m) if m.len() < offset => tail(path),
        Ok(_) => match gpu_spooler::core::read_log_from(path, offset) {
            Ok((text, offset)) => LogChunk {
                text,
                offset,
                truncated: false,
                reset: false,
                missing: false,
                path: path.display().to_string(),
            },
            Err(_) => missing(path),
        },
    }
}

fn task(state: &AppState, id: i64) -> Result<Task, String> {
    state.with_con(|con| gpu_spooler::core::get_task(con, id).map_err(|e| e.to_string()))?
}

/// (label, value) rows, as `task info` shows them.
#[tauri::command]
#[specta::specta]
pub fn task_details(state: tauri::State<'_, AppState>, id: i64) -> Result<Vec<(String, String)>, String> {
    let t = task(&state, id)?;
    let cfg = gpu_spooler::config::load_config(&state.paths).ok();
    let snapshot = gpu_spooler::core::read_gpu_snapshot(&state.paths);
    Ok(gpu_spooler::core::task_details(&t, cfg.as_ref(), snapshot.as_ref(), gpu_spooler::transport::now()))
}

/// The end of a task's log.
#[tauri::command]
#[specta::specta]
pub fn log_tail(state: tauri::State<'_, AppState>, id: i64) -> Result<LogChunk, String> {
    Ok(tail(&log_path(&task(&state, id)?, &state.paths)))
}

/// What a task's log gained since `offset`.
#[tauri::command]
#[specta::specta]
pub fn log_since(state: tauri::State<'_, AppState>, id: i64, offset: u64) -> Result<LogChunk, String> {
    Ok(since(&log_path(&task(&state, id)?, &state.paths), offset))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn follows_appends_and_restarts_after_a_rewrite() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("1.log");
        assert!(tail(&path).missing);

        std::fs::write(&path, "one\n").unwrap();
        let first = tail(&path);
        assert_eq!((first.text.as_str(), first.offset, first.reset, first.truncated), ("one\n", 4, true, false));
        assert_eq!(since(&path, first.offset).text, "");

        std::fs::OpenOptions::new().append(true).open(&path).unwrap().write_all(b"two\n").unwrap();
        let next = since(&path, first.offset);
        assert_eq!((next.text.as_str(), next.offset, next.reset), ("two\n", 8, false));

        std::fs::write(&path, "new\n").unwrap();
        let again = since(&path, next.offset);
        assert_eq!((again.text.as_str(), again.offset, again.reset), ("new\n", 4, true));

        std::fs::remove_file(&path).unwrap();
        assert!(since(&path, 4).missing);
    }

    #[test]
    fn a_long_log_is_cut_at_a_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("1.log");
        let line = "x".repeat(99) + "\n";
        std::fs::write(&path, line.repeat(3000)).unwrap();
        let t = tail(&path);
        assert!(t.truncated && t.text.starts_with('x') && t.text.len() < TAIL_BYTES as usize);
        assert_eq!(t.offset, 300_000);
    }

    #[test]
    fn log_path_defaults_to_the_logs_folder() {
        let paths = Paths::for_home("/h");
        let t = Task { id: 7, ..Default::default() };
        assert_eq!(log_path(&t, &paths), paths.logs().join("7.log"));
        let t = Task { log_file: Some("/x/7.log".into()), ..t };
        assert_eq!(log_path(&t, &paths), PathBuf::from("/x/7.log"));
    }
}
