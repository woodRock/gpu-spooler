//! Shared gpu-spooler operations behind the front ends: task queries, how a task ended, the
//! details `task info` shows, the daemon's PID file and the GPU snapshot. Nothing here prints.
//! (core.py; the operations that change the queue come with the daemon.)

use crate::config::Config;
use crate::db::{self, Task};
use crate::fmt::{fmt_duration, fmt_ts};
use crate::gpu::gpu_model;
use crate::paths::Paths;
use crate::transport::now;
use rusqlite::Connection;
use serde_json::Value;
use std::time::{Duration, Instant};

pub const ACTIVE: [&str; 2] = ["queued", "running"];
pub const FINISHED: [&str; 5] = ["done", "failed", "killed", "cancelled", "lost"];
pub const MAX_RETRIES: i64 = 3;

// Exit codes for `task wait` (see wait_exit_code).
pub const EXIT_KILLED: i32 = 130;
pub const EXIT_LOST: i32 = 75;
pub const EXIT_TIMEOUT: i32 = 124;
pub const WAIT_POLL: Duration = Duration::from_secs(2);

/// A user-facing error: unknown task, bad arguments, ...
#[derive(Debug, thiserror::Error)]
pub enum TaskError {
    #[error("{0}")]
    User(String),
    #[error("{0}")]
    Db(#[from] rusqlite::Error),
}

pub fn get_task(con: &Connection, id: i64) -> Result<Task, TaskError> {
    db::find_task(con, id)?.ok_or_else(|| TaskError::User(format!("Task {id} not found.")))
}

/// How long a task ran (so far, if running).
pub fn runtime(t: &Task) -> Option<f64> {
    let started = t.started_at.filter(|s| *s != 0.0)?;
    let ended = match t.finished_at.filter(|f| *f != 0.0) {
        Some(f) => f,
        None if t.status == "running" => now(),
        None => return None,
    };
    Some(ended - started)
}

/// How a finished task maps to a process exit code.
pub fn wait_exit_code(t: &Task) -> i32 {
    match t.status.as_str() {
        "done" => 0,
        "failed" => match t.exit_code {
            Some(0) | None => 1,
            Some(c) => c as i32,
        },
        "killed" | "cancelled" => EXIT_KILLED,
        _ => EXIT_LOST,
    }
}

/// e.g. 'task 12: done (exit 0, 3m 41s)'.
pub fn describe_end(t: &Task) -> String {
    let mut bits = vec![];
    if matches!(t.status.as_str(), "done" | "failed") {
        if let Some(c) = t.exit_code {
            bits.push(format!("exit {c}"));
        }
    }
    if t.started_at.is_some_and(|s| s != 0.0) {
        bits.push(fmt_duration(runtime(t)));
    }
    let detail = if bits.is_empty() { String::new() } else { format!(" ({})", bits.join(", ")) };
    format!("task {}: {}{detail}", t.id, t.status)
}

/// Block until every task is in a final state, reading only the database (the daemon keeps it
/// current). Returns (final rows in the order given, ids still unfinished when the timeout hit).
/// `stop` is checked between polls (Ctrl+C): when it returns true, waiting ends as a timeout would.
pub fn wait_for(
    con: &Connection,
    ids: &[i64],
    timeout: Option<Duration>,
    poll: Duration,
    stop: &dyn Fn() -> bool,
) -> Result<(Vec<Task>, Vec<i64>), TaskError> {
    for &id in ids {
        get_task(con, id)?;
    }
    let deadline = timeout.map(|t| Instant::now() + t);
    let mut done: Vec<Task> = vec![];
    loop {
        for &id in ids {
            if !done.iter().any(|t| t.id == id) {
                let t = get_task(con, id)?;
                if FINISHED.contains(&t.status.as_str()) {
                    done.push(t);
                }
            }
        }
        let pending: Vec<i64> = ids.iter().copied().filter(|id| !done.iter().any(|t| t.id == *id)).collect();
        if pending.is_empty() {
            return Ok((done, vec![]));
        }
        let pause = match deadline {
            Some(d) => {
                let left = d.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return Ok((done, pending));
                }
                left.min(poll)
            }
            None => poll,
        };
        // Sleep in short steps so `stop` is noticed quickly.
        let until = Instant::now() + pause;
        while Instant::now() < until {
            if stop() {
                return Ok((done, pending));
            }
            std::thread::sleep(until.saturating_duration_since(Instant::now()).min(Duration::from_millis(50)));
        }
    }
}

/// The daemon's latest GPU query: {"ts", "mode", "hosts": {name: ServerResult}}.
pub fn read_gpu_snapshot(paths: &Paths) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(paths.snapshot()).ok()?).ok()
}

/// GPU models a host reported in a snapshot (only these indices, if given).
pub fn snapshot_models(snapshot: Option<&Value>, host: &str, gpu_indices: Option<&str>) -> Vec<String> {
    let gpus = snapshot.and_then(|s| s.get("hosts")).and_then(|h| h.get(host)).and_then(|h| h.get("gpus"));
    let Some(gpus) = gpus.and_then(Value::as_array) else { return vec![] };
    let wanted: Vec<i64> = gpu_indices
        .unwrap_or("")
        .split(',')
        .filter(|i| crate::py::isdigit(crate::py::strip(i)))
        .filter_map(crate::py::int)
        .collect();
    let mut out: Vec<String> = vec![];
    for g in gpus {
        if !wanted.is_empty() && !g.get("index").and_then(Value::as_i64).is_some_and(|i| wanted.contains(&i)) {
            continue;
        }
        let name = g.get("name").and_then(Value::as_str).unwrap_or("");
        if !name.is_empty() {
            let m = gpu_model(name);
            if !out.contains(&m) {
                out.push(m);
            }
        }
    }
    out
}

/// (label, value) rows describing a task, for `task info` and the TUI. `cfg` is None when the
/// config can't be read (the configured GPU details are then left out).
pub fn task_details(t: &Task, cfg: Option<&Config>, snapshot: Option<&Value>, now: f64) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = vec![];
    let mut add = |label: &str, value: String| out.push((label.to_string(), value));
    add("Status", t.status.clone());
    if let Some(n) = t.name.as_ref().filter(|n| !n.is_empty()) {
        add("Label", n.clone());
    }
    add("Command", t.command.clone());
    add("GPUs req", if t.num_gpus != 0 { t.num_gpus.to_string() } else { "none (CPU job)".into() });
    let set = |v: &Option<String>| v.as_ref().filter(|s| !s.is_empty()).cloned();
    if let Some(g) = set(&t.gpu_type) {
        add("GPU type", g);
    }
    if t.min_mem != 0 {
        add("Min VRAM", format!("{} GB", t.min_mem));
    }
    if let Some(h) = set(&t.host_spec) {
        add("Hosts", h);
    }
    add("Working dir", t.working_dir.clone());
    if let Some(r) = set(&t.repo_root) {
        add("Git repo", r);
    }
    if let Some(r) = set(&t.remote_dir).filter(|r| *r != t.working_dir) {
        add("Remote dir", r);
    }
    add("Submitted", fmt_ts(Some(t.submitted_at)));
    if let Some(server) = set(&t.server) {
        let host = cfg.and_then(|c| c.hosts.get(&server));
        let models = snapshot_models(snapshot, &server, t.gpu_indices.as_deref());
        let mut shown = server.clone();
        if !models.is_empty() {
            shown += &format!("  ({})", models.join(", "));
        } else if let Some(h) = host {
            let gpu_type = h.gpu_type.clone().filter(|g| !g.is_empty());
            let mem = h.gpu_mem_gb.filter(|m| m.as_f64() != 0.0);
            if gpu_type.is_some() || mem.is_some() {
                let mem = mem.map_or("?".to_string(), |m| m.to_string());
                shown += &format!("  ({}  {mem} GB/GPU, configured)", gpu_type.unwrap_or_default());
            }
        }
        add("Server", shown);
    }
    if let Some(g) = set(&t.gpu_indices) {
        add("GPU indices", g);
    }
    if let Some(key) = set(&t.job_key) {
        let pgid = t.remote_pid.filter(|p| *p != 0).map_or(String::new(), |p| format!("  (PGID {p})"));
        add("Job", format!("~/.gpu-spooler/jobs/{key}{pgid}"));
    }
    if t.status == "running" {
        if let Some(seen) = t.last_seen.filter(|s| *s != 0.0) {
            add("Last seen", format!("{} ago", fmt_duration(Some(now - seen))));
        }
    }
    let started = t.started_at.filter(|s| *s != 0.0);
    if started.is_some() {
        add("Started", fmt_ts(started));
    }
    if let Some(f) = t.finished_at.filter(|f| *f != 0.0) {
        add("Finished", fmt_ts(Some(f)));
    }
    if started.is_some() {
        add("Runtime", fmt_duration(runtime(t)));
    }
    if let Some(c) = t.exit_code {
        add("Exit code", c.to_string());
    }
    if t.retry_count != 0 {
        add("Retries", format!("{}/{MAX_RETRIES}", t.retry_count));
    }
    if let Some(l) = set(&t.log_file) {
        add("Log file", l);
    }
    out
}

/// PID of the running daemon, or None (as core.daemon_pid: a PID that exists but belongs to
/// someone else counts as not running).
pub fn daemon_pid(paths: &Paths) -> Option<i64> {
    let text = std::fs::read_to_string(paths.pid_file()).ok()?;
    let pid = crate::py::int(&text)?;
    let pid_t = libc::pid_t::try_from(pid).ok()?;
    // SAFETY: signal 0 only checks that the process exists.
    (unsafe { libc::kill(pid_t, 0) } == 0).then_some(pid)
}

/// `getpass.getuser()`: $LOGNAME, $USER, $LNAME or $USERNAME, else the password database.
pub fn getuser() -> String {
    for var in ["LOGNAME", "USER", "LNAME", "USERNAME"] {
        if let Some(v) = std::env::var(var).ok().filter(|v| !v.is_empty()) {
            return v;
        }
    }
    // SAFETY: getpwuid returns a pointer into static storage (or null); it is read at once.
    unsafe {
        let pw = libc::getpwuid(libc::getuid());
        if !pw.is_null() && !(*pw).pw_name.is_null() {
            return std::ffi::CStr::from_ptr((*pw).pw_name).to_string_lossy().into_owned();
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(status: &str) -> Task {
        Task { id: 3, status: status.into(), ..Default::default() }
    }

    #[test]
    fn running_runtime_counts_to_now() {
        let t = Task { started_at: Some(now() - 65.0), ..task("running") };
        assert_eq!(fmt_duration(runtime(&t)), "1m 05s");
        assert_eq!(runtime(&task("queued")), None);
    }

    #[test]
    fn wait_times_out_and_stops() {
        let home = tempfile::tempdir().unwrap();
        let paths = Paths::for_home(home.path());
        let con = db::open(&paths).unwrap();
        con.execute("INSERT INTO tasks (command, submitted_at, working_dir) VALUES ('x', 1, '/')", []).unwrap();
        con.execute("INSERT INTO tasks (command, submitted_at, working_dir, status) VALUES ('y', 1, '/', 'done')", [])
            .unwrap();
        let (done, pending) =
            wait_for(&con, &[1, 2], Some(Duration::from_millis(100)), Duration::from_millis(20), &|| false).unwrap();
        assert_eq!((done.iter().map(|t| t.id).collect::<Vec<_>>(), pending), (vec![2], vec![1]));
        let (_, pending) = wait_for(&con, &[1], None, Duration::from_secs(5), &|| true).unwrap();
        assert_eq!(pending, [1]);
        assert_eq!(wait_for(&con, &[9], None, WAIT_POLL, &|| false).unwrap_err().to_string(), "Task 9 not found.");
    }

    #[test]
    fn daemon_pid_checks_the_process() {
        let home = tempfile::tempdir().unwrap();
        let paths = Paths::for_home(home.path());
        assert_eq!(daemon_pid(&paths), None);
        std::fs::create_dir_all(&paths.state).unwrap();
        std::fs::write(paths.pid_file(), format!("{}\n", std::process::id())).unwrap();
        assert_eq!(daemon_pid(&paths), Some(std::process::id() as i64));
        std::fs::write(paths.pid_file(), "junk").unwrap();
        assert_eq!(daemon_pid(&paths), None);
    }
}
