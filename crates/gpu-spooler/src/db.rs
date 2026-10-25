//! The queue database, ~/.gpu-spooler/tasks.db, shared with the Python version: the same schema,
//! the same append-only column migrations, WAL journal. Both versions can open it at once, and a
//! row written by either reads the same in the other.

use crate::paths::Paths;
use rusqlite::types::ValueRef;
use rusqlite::{Connection, Row};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    command      TEXT    NOT NULL,
    num_gpus     INTEGER NOT NULL DEFAULT 1,
    min_mem      INTEGER NOT NULL DEFAULT 0,
    status       TEXT    NOT NULL DEFAULT 'queued',
    server       TEXT,
    gpu_indices  TEXT,
    ssh_pid      INTEGER,
    submitted_at REAL    NOT NULL,
    started_at   REAL,
    finished_at  REAL,
    exit_code    INTEGER,
    log_file     TEXT,
    working_dir  TEXT    NOT NULL,
    name         TEXT,
    retry_count  INTEGER NOT NULL DEFAULT 0
);
";

/// Columns added after the original schema: (name, definition). Append only, and keep in step
/// with core.py's _MIGRATIONS while both versions are in use.
const MIGRATIONS: &[(&str, &str)] = &[
    ("retry_count", "INTEGER NOT NULL DEFAULT 0"),
    ("min_mem", "INTEGER NOT NULL DEFAULT 0"),
    ("job_key", "TEXT"),
    ("remote_pid", "INTEGER"),
    ("log_offset", "INTEGER NOT NULL DEFAULT 0"),
    ("host_spec", "TEXT"),
    ("remote_dir", "TEXT"),
    ("last_seen", "REAL"),
    ("repo_root", "TEXT"),                       // local git repo the task was submitted from
    ("gpu_type", "TEXT"),                        // --gpu-type: nvidia-smi model name(s), comma separated
    ("git_commit", "TEXT"),                      // the commit the server ran, captured at launch (run provenance)
    ("git_dirty", "INTEGER"),                    // uncommitted changes there at launch
    ("git_dirty_paths", "TEXT"),                 // which tracked files (newline separated, at most 50)
    ("no_record", "INTEGER NOT NULL DEFAULT 0"), // task submit --no-record: not a taura run
];

/// One row of `tasks`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Task {
    pub id: i64,
    pub command: String,
    pub num_gpus: i64,
    pub min_mem: i64,
    pub status: String,
    pub server: Option<String>,
    pub gpu_indices: Option<String>,
    pub ssh_pid: Option<i64>,
    pub submitted_at: f64,
    pub started_at: Option<f64>,
    pub finished_at: Option<f64>,
    pub exit_code: Option<i64>,
    pub log_file: Option<String>,
    pub working_dir: String,
    pub name: Option<String>,
    pub retry_count: i64,
    pub job_key: Option<String>,
    pub remote_pid: Option<i64>,
    pub log_offset: i64,
    pub host_spec: Option<String>,
    pub remote_dir: Option<String>,
    pub last_seen: Option<f64>,
    pub repo_root: Option<String>,
    pub gpu_type: Option<String>,
    pub git_commit: Option<String>,
    pub git_dirty: Option<i64>,
    pub git_dirty_paths: Option<String>,
    pub no_record: i64,
}

// SQLite keeps whatever type was written, so read leniently: a number where text is expected
// is shown as Python would show it, and text where a number is expected counts as unset.
fn text(row: &Row, col: &str) -> rusqlite::Result<Option<String>> {
    Ok(match row.get_ref(col)? {
        ValueRef::Null => None,
        ValueRef::Text(t) | ValueRef::Blob(t) => Some(String::from_utf8_lossy(t).into_owned()),
        ValueRef::Integer(i) => Some(i.to_string()),
        ValueRef::Real(f) => Some(crate::py::repr_float(f)),
    })
}

fn int(row: &Row, col: &str) -> rusqlite::Result<Option<i64>> {
    Ok(match row.get_ref(col)? {
        ValueRef::Integer(i) => Some(i),
        ValueRef::Real(f) => Some(f as i64),
        ValueRef::Text(t) => std::str::from_utf8(t).ok().and_then(crate::py::int),
        _ => None,
    })
}

fn real(row: &Row, col: &str) -> rusqlite::Result<Option<f64>> {
    Ok(match row.get_ref(col)? {
        ValueRef::Integer(i) => Some(i as f64),
        ValueRef::Real(f) => Some(f),
        ValueRef::Text(t) => std::str::from_utf8(t).ok().and_then(|s| s.trim().parse().ok()),
        _ => None,
    })
}

impl Task {
    pub fn from_row(row: &Row) -> rusqlite::Result<Task> {
        Ok(Task {
            id: int(row, "id")?.unwrap_or(0),
            command: text(row, "command")?.unwrap_or_default(),
            num_gpus: int(row, "num_gpus")?.unwrap_or(0),
            min_mem: int(row, "min_mem")?.unwrap_or(0),
            status: text(row, "status")?.unwrap_or_default(),
            server: text(row, "server")?,
            gpu_indices: text(row, "gpu_indices")?,
            ssh_pid: int(row, "ssh_pid")?,
            submitted_at: real(row, "submitted_at")?.unwrap_or(0.0),
            started_at: real(row, "started_at")?,
            finished_at: real(row, "finished_at")?,
            exit_code: int(row, "exit_code")?,
            log_file: text(row, "log_file")?,
            working_dir: text(row, "working_dir")?.unwrap_or_default(),
            name: text(row, "name")?,
            retry_count: int(row, "retry_count")?.unwrap_or(0),
            job_key: text(row, "job_key")?,
            remote_pid: int(row, "remote_pid")?,
            log_offset: int(row, "log_offset")?.unwrap_or(0),
            host_spec: text(row, "host_spec")?,
            remote_dir: text(row, "remote_dir")?,
            last_seen: real(row, "last_seen")?,
            repo_root: text(row, "repo_root")?,
            gpu_type: text(row, "gpu_type")?,
            git_commit: text(row, "git_commit")?,
            git_dirty: int(row, "git_dirty")?,
            git_dirty_paths: text(row, "git_dirty_paths")?,
            no_record: int(row, "no_record")?.unwrap_or(0),
        })
    }
}

/// Open (creating if needed) the queue database, and the logs folder next to it.
pub fn open(paths: &Paths) -> rusqlite::Result<Connection> {
    let mkdir =
        |p: &std::path::Path| std::fs::create_dir_all(p).map_err(|e| rusqlite::Error::ToSqlConversionFailure(e.into()));
    mkdir(&paths.state)?;
    mkdir(&paths.logs())?;
    let con = Connection::open(paths.db())?;
    con.busy_timeout(Duration::from_secs(10))?;
    con.query_row("PRAGMA journal_mode=WAL", [], |_| Ok(()))?;
    con.execute_batch(SCHEMA)?;
    let cols: Vec<String> =
        con.prepare("PRAGMA table_info(tasks)")?.query_map([], |r| r.get(1))?.collect::<Result<_, _>>()?;
    for (name, definition) in MIGRATIONS {
        if !cols.iter().any(|c| c == name) {
            con.execute_batch(&format!("ALTER TABLE tasks ADD COLUMN {name} {definition}"))?;
        }
    }
    Ok(con)
}

/// Most recent tasks first.
pub fn list_tasks(con: &Connection, limit: Option<usize>) -> rusqlite::Result<Vec<Task>> {
    let sql = match limit {
        None => "SELECT * FROM tasks ORDER BY id DESC".to_string(),
        Some(n) => format!("SELECT * FROM tasks ORDER BY id DESC LIMIT {n}"),
    };
    con.prepare(&sql)?.query_map([], Task::from_row)?.collect()
}

/// A task by id, if it exists.
pub fn find_task(con: &Connection, id: i64) -> rusqlite::Result<Option<Task>> {
    let mut stmt = con.prepare("SELECT * FROM tasks WHERE id=?")?;
    let mut rows = stmt.query_map([id], Task::from_row)?;
    rows.next().transpose()
}

/// Task counts: "total", plus one per status ("running" and "queued" always present).
pub fn task_counts(con: &Connection) -> rusqlite::Result<std::collections::BTreeMap<String, i64>> {
    let mut counts: std::collections::BTreeMap<String, i64> =
        [("total".to_string(), 0), ("running".to_string(), 0), ("queued".to_string(), 0)].into();
    let mut stmt = con.prepare("SELECT status, COUNT(*) FROM tasks GROUP BY status")?;
    for row in stmt.query_map([], |r| Ok((r.get::<_, Option<String>>(0)?.unwrap_or_default(), r.get::<_, i64>(1)?)))? {
        let (status, n) = row?;
        *counts.get_mut("total").unwrap() += n;
        counts.insert(status, n);
    }
    Ok(counts)
}

/// GPUs held by running tasks (what [limits] max_gpus counts).
pub fn gpus_in_use(con: &Connection) -> rusqlite::Result<i64> {
    con.query_row("SELECT COALESCE(SUM(num_gpus), 0) FROM tasks WHERE status='running'", [], |r| r.get(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_migrates_and_reads_rows() {
        let home = tempfile::tempdir().unwrap();
        let paths = Paths::for_home(home.path());
        // An old database, from before the migrations: open() adds the missing columns.
        std::fs::create_dir_all(&paths.state).unwrap();
        let old = Connection::open(paths.db()).unwrap();
        old.execute_batch(SCHEMA).unwrap();
        old.execute(
            "INSERT INTO tasks (command, submitted_at, working_dir, gpu_indices, started_at) VALUES ('x', 5, '/w', 0, '7.5')",
            [],
        )
        .unwrap();
        drop(old);
        let con = open(&paths).unwrap();
        assert!(paths.logs().is_dir());
        let t = find_task(&con, 1).unwrap().unwrap();
        assert_eq!((t.command.as_str(), t.submitted_at, t.status.as_str()), ("x", 5.0, "queued"));
        assert_eq!((t.gpu_indices.as_deref(), t.started_at, t.log_offset, t.job_key), (Some("0"), Some(7.5), 0, None));
        assert!(find_task(&con, 2).unwrap().is_none());
        con.execute("INSERT INTO tasks (command, submitted_at, working_dir, status, num_gpus) VALUES ('y', 6, '/w', 'running', 2)", [])
            .unwrap();
        assert_eq!(list_tasks(&con, Some(1)).unwrap()[0].id, 2);
        assert_eq!(list_tasks(&con, None).unwrap().len(), 2);
        let counts = task_counts(&con).unwrap();
        assert_eq!((counts["total"], counts["running"], counts["queued"]), (2, 1, 1));
        assert_eq!(gpus_in_use(&con).unwrap(), 2);
        // Opening again is a no-op.
        drop(con);
        open(&paths).unwrap();
    }
}
