//! Shared gpu-spooler operations behind the front ends: task queries, how a task ended, the
//! details `task info` shows, the daemon's PID file and the GPU snapshot. Nothing here prints.
//! (core.py; the operations that change the queue come with the daemon.)

use crate::config::Config;
use crate::config::Host;
use crate::db::{self, Task};
use crate::fmt::{fmt_duration, fmt_ts};
use crate::gpu::gpu_model;
use crate::paths::Paths;
use crate::transport::{now, Transport};
use rusqlite::Connection;
use serde_json::Value;
use std::collections::HashSet;
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
    #[error("{0}")]
    Io(#[from] std::io::Error),
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

// ── Changing the queue ────────────────────────────────────────────────────────

/// Fields reset when a task goes back to the queue for another attempt.
pub const REQUEUE: &str = "status='queued', ssh_pid=NULL, started_at=NULL, finished_at=NULL, server=NULL, \
gpu_indices=NULL, job_key=NULL, remote_pid=NULL, log_offset=0, remote_dir=NULL, retry_count=retry_count+1";

/// What `task submit` asks for.
#[derive(Clone, Debug, Default)]
pub struct SubmitOpts {
    pub command: String,
    pub num_gpus: i64,
    pub min_mem: i64,
    pub name: Option<String>,
    pub host_spec: Option<String>,
    pub working_dir: Option<String>,
    pub gpu_type: Option<String>,
    pub no_record: bool,
    /// Start the daemon (unless it runs).
    pub start: bool,
    /// Check the servers' copy of the repo (one ssh call per copy).
    pub check_repo: bool,
}

#[derive(Clone, Debug)]
pub struct Submitted {
    pub task_id: i64,
    pub command: String,
    pub working_dir: String,
    pub num_gpus: i64,
    pub min_mem: i64,
    pub name: Option<String>,
    pub host_spec: Option<String>,
    /// The hosts the task may run on.
    pub candidates: Vec<String>,
    pub warnings: Vec<String>,
    pub gpu_type: Option<String>,
}

/// The checks submit makes before it needs the config. Returns the --gpu-type, tidied.
pub fn check_submit(num_gpus: i64, command: &str, gpu_type: Option<&str>) -> Result<Option<String>, TaskError> {
    if num_gpus < 0 {
        return Err(TaskError::User("-G must be 0 (CPU job) or more".into()));
    }
    if crate::py::strip(command).is_empty() {
        return Err(TaskError::User("empty command".into()));
    }
    let types: Vec<&str> = gpu_type.unwrap_or("").split(',').map(crate::py::strip).filter(|t| !t.is_empty()).collect();
    let gpu_type = (!types.is_empty()).then(|| types.join(","));
    if gpu_type.is_some() && num_gpus == 0 {
        return Err(TaskError::User("--gpu-type needs a GPU job (-G 1 or more)".into()));
    }
    Ok(gpu_type)
}

fn hostname() -> String {
    let mut buf = [0u8; 256];
    // SAFETY: gethostname writes at most buf.len() bytes into buf.
    if unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) } != 0 {
        return String::new();
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// Best guess at whether `path` is visible as-is on the ECS servers.
pub fn on_ecs_filesystem(path: &str) -> bool {
    hostname().ends_with(".vuw.ac.nz")
        || path.starts_with("/vol/")
        || path.starts_with("/am/")
        || (cfg!(target_os = "linux") && std::path::Path::new("/vol").is_dir())
}

/// The cwd as the shell spells it ($PWD keeps symlinks, like `pwd -L`).
pub fn logical_cwd() -> String {
    use std::os::unix::fs::MetadataExt;
    let cwd = std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_else(|_| "/".into());
    if let Ok(pwd) = std::env::var("PWD") {
        if pwd.starts_with('/') {
            let same = match (std::fs::metadata(&pwd), std::fs::metadata(&cwd)) {
                (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
                _ => false,
            };
            if same {
                return crate::config::normpath(&pwd);
            }
        }
    }
    cwd
}

/// Queue a task (and start the daemon unless `opts.start` is false). When submitted from a git
/// repo, the remote copies are checked and any that are missing or stale produce a warning.
pub fn submit(transport: &Transport, opts: &SubmitOpts) -> Result<Submitted, TaskError> {
    let gpu_type = check_submit(opts.num_gpus, &opts.command, opts.gpu_type.as_deref())?;
    let cfg = &transport.cfg;
    if let Some(cap) = cfg.max_gpus.filter(|cap| opts.num_gpus > *cap) {
        return Err(TaskError::User(format!(
            "-G {} is more than [limits] max_gpus = {cap}; the task could never start",
            opts.num_gpus
        )));
    }
    let host_spec = opts.host_spec.clone().filter(|h| !h.is_empty());
    let candidates = cfg
        .select(host_spec.as_deref())
        .map_err(|e| TaskError::User(format!("-H {}: {e}", host_spec.as_deref().unwrap_or("None"))))?;
    let working_dir = match opts.working_dir.as_deref().filter(|d| !d.is_empty()) {
        Some(d) => crate::config::abspath(&transport.paths.expand_user(d)),
        None => logical_cwd(),
    };

    // Off an ECS filesystem, a git repo is run from its remote copy (see repo.rs).
    let on_ecs = on_ecs_filesystem(&working_dir);
    let local_repo = if on_ecs { None } else { crate::repo::find_repo(&working_dir) };

    let con = db::open(&transport.paths)?;
    con.execute(
        "INSERT INTO tasks (command, num_gpus, min_mem, submitted_at, working_dir, name, host_spec, repo_root, \
         gpu_type, no_record) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rusqlite::params![
            opts.command,
            opts.num_gpus,
            opts.min_mem,
            now(),
            working_dir,
            opts.name,
            host_spec,
            local_repo.as_ref().map(|r| r.root.clone()),
            gpu_type,
            opts.no_record as i64
        ],
    )?;
    let mut sub = Submitted {
        task_id: con.last_insert_rowid(),
        command: opts.command.clone(),
        working_dir: working_dir.clone(),
        num_gpus: opts.num_gpus,
        min_mem: opts.min_mem,
        name: opts.name.clone(),
        host_spec,
        candidates: candidates.iter().map(|h| h.name.clone()).collect(),
        warnings: vec![],
        gpu_type: gpu_type.clone(),
    };
    if let Some(gt) = &gpu_type {
        if let Some(problem) = gpu_type_problem(&transport.paths, gt, &sub.candidates) {
            sub.warnings.push(problem);
        }
    }
    if opts.num_gpus == 0 && !candidates.iter().any(|h| h.cpu_slots > 0) {
        sub.warnings.push("no selected host has cpu_slots > 0; this task will stay queued (see `task hosts`).".into());
    }
    let copies = match &local_repo {
        Some(r) => crate::repo::targets(cfg, &candidates, r, &working_dir),
        None => vec![],
    };
    let covered: HashSet<&str> = copies.iter().flat_map(|t| t.members.iter().map(String::as_str)).collect();
    let unmapped: Vec<&str> = candidates
        .iter()
        .filter(|h| !h.is_local() && !covered.contains(h.name.as_str()) && !cfg.is_mapped(h, &working_dir))
        .map(|h| h.name.as_str())
        .collect();
    if !unmapped.is_empty() && !on_ecs {
        sub.warnings.push(format!(
            "{working_dir} has no path_map entry for {}; it will be used unchanged there.",
            crate::fmt::truncate(&unmapped.join(", "), 40)
        ));
    }
    if opts.check_repo && !copies.is_empty() {
        let repo = local_repo.as_ref().expect("copies come from a repo");
        sub.warnings.extend(repo_status(transport, repo, &copies).into_iter().filter_map(|(_, w)| w));
    }
    if opts.start {
        start_daemon(&transport.paths)?;
    }
    Ok(sub)
}

/// (target, problem or None) for each remote copy, checked in parallel.
pub fn repo_status<'c>(
    transport: &Transport,
    repo: &crate::repo::LocalRepo,
    copies: &[crate::repo::Target<'c>],
) -> Vec<(crate::repo::Target<'c>, Option<String>)> {
    let hosts: Vec<&Host> = copies.iter().map(|t| t.host).collect();
    let states = transport.fan_out(
        &hosts,
        |h| {
            let t = copies.iter().find(|t| t.host.name == h.name).expect("one copy per host");
            crate::repo::check(transport, t)
        },
        16,
    );
    copies.iter().map(|t| (t.clone(), crate::repo::staleness(repo, t, &states[&t.host.name]))).collect()
}

/// A warning if no candidate host reported a GPU of this type in the latest snapshot (None if
/// one did, or if there is no snapshot to judge by).
pub fn gpu_type_problem(paths: &Paths, gpu_type: &str, candidates: &[String]) -> Option<String> {
    let snapshot = read_gpu_snapshot(paths);
    let hosts = snapshot.as_ref().and_then(|s| s.get("hosts")).and_then(Value::as_object)?;
    let seen: Vec<Vec<String>> = candidates
        .iter()
        .filter(|n| hosts.contains_key(*n))
        .map(|n| snapshot_models(snapshot.as_ref(), n, None))
        .collect();
    if seen.iter().all(Vec::is_empty) {
        return None;
    }
    if seen.iter().flatten().any(|m| crate::gpu::gpu_type_matches(Some(gpu_type), m)) {
        return None;
    }
    let known: std::collections::BTreeSet<&String> = seen.iter().flatten().collect();
    Some(format!(
        "no selected host reported a GPU of type '{gpu_type}' in the last GPU poll (seen: {}); the task will stay \
         queued unless one does.",
        known.into_iter().cloned().collect::<Vec<_>>().join(", ")
    ))
}

/// Queue a copy of an existing task (same command, resources and directory).
pub fn resubmit(transport: &Transport, task_id: i64) -> Result<Submitted, TaskError> {
    let t = get_task(&db::open(&transport.paths)?, task_id)?;
    submit(
        transport,
        &SubmitOpts {
            command: t.command,
            num_gpus: t.num_gpus,
            min_mem: t.min_mem,
            name: t.name,
            host_spec: t.host_spec,
            working_dir: Some(t.working_dir),
            gpu_type: t.gpu_type,
            no_record: t.no_record != 0,
            start: true,
            check_repo: true,
        },
    )
}

#[derive(Clone, Debug)]
pub struct KillResult {
    pub task_id: i64,
    /// cancelled | killed | none
    pub action: &'static str,
    /// false: the host could not be reached
    pub confirmed: bool,
    pub message: String,
}

/// Signal a running task's process group on its host: (confirmed, message), not confirmed if the
/// host was unreachable.
fn kill_remote(transport: &Transport, t: &Task) -> (bool, String) {
    let Some(key) = t.job_key.as_ref().filter(|k| !k.is_empty()) else {
        // Started by the old version: the job lives inside a local ssh client.
        return match t.ssh_pid.filter(|p| *p != 0) {
            Some(pid) => {
                // SAFETY: sending SIGTERM to a recorded PID.
                if unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) } == 0 {
                    (true, format!("Sent SIGTERM to SSH client (PID {pid})."))
                } else {
                    (true, "SSH process already gone.".into())
                }
            }
            None => (true, "No process recorded.".into()),
        };
    };
    let server = t.server.clone().unwrap_or_default();
    let Some(host) = transport.cfg.hosts.get(&server) else {
        return (false, format!("host '{}' is not in the config", t.server.as_deref().unwrap_or("None")));
    };
    let r = transport.run(host, &crate::jobs::kill_script(key, 10), Duration::from_secs(30));
    if r.rc == 0 && r.stdout.contains("KILLED") {
        let pgid = t.remote_pid.filter(|p| *p != 0).map_or("?".to_string(), |p| p.to_string());
        return (
            true,
            format!(
                "Sent SIGTERM to process group {pgid} on {} (SIGKILL follows in 10s if it is still alive).",
                host.name
            ),
        );
    }
    if r.rc == 0 && (r.stdout.contains("NOT_RUNNING") || r.stdout.contains("NO_JOB")) {
        return (true, format!("Process on {} was already gone.", host.name));
    }
    let err = crate::py::strip(&r.stderr);
    let reason = crate::py::splitlines(err)
        .last()
        .filter(|_| !err.is_empty())
        .map_or(format!("exit {}", r.rc), |l| l.to_string());
    let tail = match t.remote_pid.filter(|p| *p != 0) {
        Some(pid) => format!(" and `kill -- -{pid}`."),
        None => ".".into(),
    };
    (
        false,
        format!(
            "could not reach {}: {reason}. The process may still be running; check with `task ssh {server}`{tail}",
            host.name
        ),
    )
}

/// Cancel a queued task or kill a running one.
pub fn kill(transport: &Transport, task_id: i64) -> Result<KillResult, TaskError> {
    let con = db::open(&transport.paths)?;
    let mut t = get_task(&con, task_id)?;
    if t.status == "queued" {
        let n = con.execute(
            "UPDATE tasks SET status='cancelled', finished_at=? WHERE id=? AND status='queued'",
            rusqlite::params![now(), task_id],
        )?;
        if n > 0 {
            return Ok(KillResult {
                task_id,
                action: "cancelled",
                confirmed: true,
                message: format!("Task {task_id} cancelled."),
            });
        }
        t = get_task(&con, task_id)?; // it started in the meantime
    }
    if t.status != "running" {
        return Ok(KillResult {
            task_id,
            action: "none",
            confirmed: true,
            message: format!("Task {task_id} is already {}.", t.status),
        });
    }
    // Mark killed first so the daemon doesn't treat the signal as a failure and retry it.
    con.execute(
        "UPDATE tasks SET status='killed', finished_at=?, ssh_pid=NULL WHERE id=?",
        rusqlite::params![now(), task_id],
    )?;
    let (confirmed, message) = kill_remote(transport, &t);
    Ok(KillResult { task_id, action: "killed", confirmed, message })
}

pub struct KillAllResult {
    pub cancelled: usize,
    pub killed: usize,
    pub unconfirmed: Vec<KillResult>,
}

/// `f` over `items` on up to `workers` threads, results in input order.
fn parallel_map<T: Sync, R: Send>(items: &[T], workers: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<R>>> = items.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..workers.clamp(1, items.len().max(1)) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let Some(item) = items.get(i) else { break };
                *slots[i].lock().unwrap() = Some(f(item));
            });
        }
    });
    slots.into_iter().map(|s| s.into_inner().unwrap().expect("every item ran")).collect()
}

/// Cancel every queued task and kill every running one, in parallel.
pub fn kill_all(transport: &Transport) -> Result<KillAllResult, TaskError> {
    let con = db::open(&transport.paths)?;
    let now = now();
    let cancelled = con.execute("UPDATE tasks SET status='cancelled', finished_at=? WHERE status='queued'", [now])?;
    let running: Vec<Task> = con
        .prepare("SELECT * FROM tasks WHERE status='running'")?
        .query_map([], Task::from_row)?
        .collect::<Result<_, _>>()?;
    let killed =
        con.execute("UPDATE tasks SET status='killed', finished_at=?, ssh_pid=NULL WHERE status='running'", [now])?;

    let cfg = &transport.cfg;
    let hosts: Vec<&Host> = running.iter().filter_map(|t| t.server.as_ref().and_then(|s| cfg.hosts.get(s))).collect();
    transport.warm_up(&hosts);
    let results = parallel_map(&running, 16, |t| kill_remote(transport, t));
    let unconfirmed = running
        .iter()
        .zip(results)
        .filter(|(_, (ok, _))| !ok)
        .map(|(t, (_, message))| KillResult { task_id: t.id, action: "killed", confirmed: false, message })
        .collect();
    Ok(KillAllResult { cancelled, killed, unconfirmed })
}

/// Delete every task and restart IDs at 1.
pub fn reset_history(paths: &Paths) -> Result<(), TaskError> {
    let con = db::open(paths)?;
    con.execute_batch("DELETE FROM tasks; DELETE FROM sqlite_sequence WHERE name='tasks';")?;
    Ok(())
}

pub struct Cleared {
    pub tasks: usize,
    pub log_files: usize,
    pub failed_hosts: Vec<String>,
}

/// Remove finished tasks, their local logs and their job directories.
pub fn clear_finished(transport: &Transport) -> Result<Cleared, TaskError> {
    let con = db::open(&transport.paths)?;
    let finished = format!("status IN ({})", FINISHED.map(|s| format!("'{s}'")).join(","));
    let to_delete: Vec<(i64, Option<String>, Option<String>)> = con
        .prepare(&format!("SELECT id, server, log_file FROM tasks WHERE {finished}"))?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<Result<_, _>>()?;
    let tasks = con.execute(&format!("DELETE FROM tasks WHERE {finished}"), [])?;

    let log_files = to_delete
        .iter()
        .filter(|(_, _, log)| log.as_ref().is_some_and(|l| !l.is_empty() && std::fs::remove_file(l).is_ok()))
        .count();

    // Remove job directories on the hosts (best effort).
    let cfg = &transport.cfg;
    let mut by_host: indexmap::IndexMap<String, Vec<i64>> = indexmap::IndexMap::new();
    for (id, server, _) in &to_delete {
        if let Some(s) = server.as_ref().filter(|s| cfg.hosts.contains_key(*s)) {
            by_host.entry(s.clone()).or_default().push(*id);
        }
    }
    let hosts: Vec<&Host> = by_host.keys().map(|h| &cfg.hosts[h]).collect();
    let cleaned = transport.fan_out(
        &hosts,
        |h| transport.run(h, &crate::jobs::cleanup_script(&by_host[&h.name]), Duration::from_secs(30)).rc == 0,
        16,
    );
    let mut failed_hosts: Vec<String> = cleaned.into_iter().filter(|(_, ok)| !ok).map(|(h, _)| h).collect();
    failed_hosts.sort();
    Ok(Cleared { tasks, log_files, failed_hosts })
}

// ── The daemon process ────────────────────────────────────────────────────────

/// Start the daemon (`task __daemon`, detached in its own session, output to daemon.log); its
/// PID, or None if one is already running.
pub fn start_daemon(paths: &Paths) -> std::io::Result<Option<i64>> {
    use std::os::unix::process::CommandExt;
    if daemon_pid(paths).is_some() {
        return Ok(None);
    }
    std::fs::create_dir_all(&paths.state)?;
    let log = std::fs::OpenOptions::new().create(true).append(true).open(paths.daemon_log())?;
    let mut cmd = std::process::Command::new(std::env::current_exe()?);
    cmd.arg("__daemon")
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .current_dir(&paths.home);
    // SAFETY: setsid is async-signal-safe; it detaches the daemon from this terminal.
    unsafe {
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    let child = cmd.spawn()?;
    let pid = child.id() as i64;
    std::fs::write(paths.pid_file(), pid.to_string())?;
    Ok(Some(pid))
}

/// Stop the daemon; the PID it had, or None if it wasn't running.
pub fn stop_daemon(paths: &Paths) -> Option<i64> {
    let pid = daemon_pid(paths);
    let _ = std::fs::remove_file(paths.pid_file());
    let pid = pid?;
    // SAFETY: SIGTERM to the daemon's PID.
    (unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) } == 0).then_some(pid)
}

/// Ask the daemon to keep the GPU snapshot fresh for the next couple of minutes.
pub fn request_gpu_snapshot(paths: &Paths) -> std::io::Result<()> {
    std::fs::create_dir_all(&paths.state)?;
    let f = std::fs::OpenOptions::new().create(true).append(true).open(paths.snapshot_request())?;
    f.set_modified(std::time::SystemTime::now())
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
