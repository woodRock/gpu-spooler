//! The queue as the window shows it: one read of the database, the daemon's PID file and the GPU
//! cap, sent to the frontend whole whenever it changes.

use gpu_spooler::config::Config;
use gpu_spooler::db::{self, Task};
use gpu_spooler::paths::Paths;
use rusqlite::Connection;
use serde::Serialize;

/// Tasks the window lists, newest first (`task list` shows the same 200).
pub const LIST_LIMIT: usize = 200;

/// Everything the queue view needs.
#[derive(Clone, Debug, PartialEq, Serialize, specta::Type, tauri_specta::Event)]
pub struct QueueState {
    pub tasks: Vec<Task>,
    /// None: the daemon is not running.
    pub daemon_pid: Option<i64>,
    /// GPUs held by running tasks.
    pub gpus_in_use: i64,
    /// `[limits] max_gpus`, if set.
    pub max_gpus: Option<i64>,
    /// Why the read failed (the previous tasks are kept); None when it worked.
    pub error: Option<String>,
}

/// Read the queue through an open connection. A config error (`cfg` Err) does not stop the read:
/// the cap is then unknown, and the error is reported.
pub fn read(con: &Connection, paths: &Paths, cfg: Result<&Config, &str>) -> QueueState {
    let daemon_pid = gpu_spooler::core::daemon_pid(paths);
    let (max_gpus, config_error) = match cfg {
        Ok(cfg) => (cfg.max_gpus, None),
        Err(e) => (None, Some(format!("config: {e}"))),
    };
    let tasks = db::list_tasks(con, Some(LIST_LIMIT)).and_then(|t| Ok((t, db::gpus_in_use(con)?)));
    match tasks {
        Ok((tasks, gpus_in_use)) => QueueState { tasks, daemon_pid, gpus_in_use, max_gpus, error: config_error },
        Err(e) => {
            QueueState { tasks: vec![], daemon_pid, gpus_in_use: 0, max_gpus, error: Some(format!("database: {e}")) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_tasks_gpus_and_daemon() {
        let home = tempfile::tempdir().unwrap();
        let paths = Paths::for_home(home.path());
        let con = db::open(&paths).unwrap();
        con.execute(
            "INSERT INTO tasks (command, submitted_at, working_dir, status, num_gpus) VALUES ('a', 1, '/', 'running', 2)",
            [],
        )
        .unwrap();
        con.execute("INSERT INTO tasks (command, submitted_at, working_dir) VALUES ('b', 2, '/')", []).unwrap();
        let s = read(&con, &paths, Ok(&gpu_spooler::config::load_config(&paths).unwrap()));
        assert_eq!(s.tasks.iter().map(|t| t.command.as_str()).collect::<Vec<_>>(), ["b", "a"]);
        assert_eq!((s.gpus_in_use, s.daemon_pid, s.error), (2, None, None));

        std::fs::write(paths.pid_file(), std::process::id().to_string()).unwrap();
        assert_eq!(read(&con, &paths, Err("x")).daemon_pid, Some(std::process::id() as i64));
    }

    #[test]
    fn a_bad_config_still_reads_the_queue() {
        let home = tempfile::tempdir().unwrap();
        let paths = Paths::for_home(home.path());
        let con = db::open(&paths).unwrap();
        con.execute("INSERT INTO tasks (command, submitted_at, working_dir) VALUES ('b', 2, '/')", []).unwrap();
        let s = read(&con, &paths, Err("bad toml"));
        assert_eq!((s.tasks.len(), s.max_gpus, s.error.as_deref()), (1, None, Some("config: bad toml")));
    }
}
