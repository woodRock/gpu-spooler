//! The gpu-spooler daemon: dispatches queued tasks to hosts and polls detached jobs for
//! completion, syncing their logs. Started by [`crate::core::start_daemon`] (`task __daemon`).
//!
//! All job state lives in the database and the job directories, so a restarted daemon re-adopts
//! jobs that are still running, whichever version (Rust or Python) started them. Log messages
//! are the Python daemon's word for word.

use crate::config::{load_config, Config, Host};
use crate::core::{MAX_RETRIES, REQUEUE};
use crate::db::{self, Task};
use crate::gpu::{gpu_type_matches, query_hosts, write_snapshot, ServerResult};
use crate::jobs::{self, JobStatus};
use crate::paths::Paths;
use crate::transport::{auth_hint, now, Mode, Transport};
use crate::{provenance, py};
use indexmap::IndexMap;
use rusqlite::types::Value as Sql;
use rusqlite::{params_from_iter, Connection};
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

/// Seconds between GPU polls / dispatch.
pub const POLL_INTERVAL: f64 = 30.0;
pub const LAUNCH_TIMEOUT: Duration = Duration::from_secs(60);
pub const STATUS_TIMEOUT: Duration = Duration::from_secs(60);
/// Keep refreshing the GPU snapshot this long after a request.
pub const SNAPSHOT_WANTED: f64 = 120.0;

/// Seconds between status / log-sync polls of running jobs (`GPU_SPOOLER_STATUS_INTERVAL`).
pub fn status_interval() -> f64 {
    std::env::var("GPU_SPOOLER_STATUS_INTERVAL").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(5.0)
}

// ── Assignment ────────────────────────────────────────────────────────────────

/// What a host can take right now, for one dispatch round.
#[derive(Clone, Debug)]
pub struct HostState<'c> {
    pub host: &'c Host,
    /// gpu index → free, in index order
    pub free: BTreeMap<i64, bool>,
    /// unused CPU slots
    pub cpu_free: i64,
    /// smallest detected GPU memory (decimal GB)
    pub detected_mem_gb: f64,
    /// gpu index → nvidia-smi name
    pub names: BTreeMap<i64, String>,
}

impl HostState<'_> {
    /// Detected memory when the host reported GPUs; the configured value otherwise.
    pub fn mem_gb(&self) -> f64 {
        match self.host.gpu_mem_gb {
            Some(m) if self.detected_mem_gb == 0.0 => m.as_f64(),
            _ => self.detected_mem_gb,
        }
    }
}

/// The GPUs, CPU slots and server of a running task: what a dispatch round counts as taken.
#[derive(Clone, Debug, Default)]
pub struct Running {
    pub server: Option<String>,
    pub gpu_indices: Option<String>,
    pub num_gpus: i64,
}

impl From<&Task> for Running {
    fn from(t: &Task) -> Running {
        Running { server: t.server.clone(), gpu_indices: t.gpu_indices.clone(), num_gpus: t.num_gpus }
    }
}

/// GPUs our tasks may still take under [limits] max_gpus (None: no limit).
pub fn gpu_budget(cfg: &Config, running: &[Running]) -> Option<i64> {
    cfg.max_gpus.map(|m| (m - running.iter().map(|r| r.num_gpus).sum::<i64>()).max(0))
}

/// Per-host availability, treating our own running tasks as busy (results order kept).
pub fn host_states<'c>(
    cfg: &'c Config,
    results: &IndexMap<String, ServerResult>,
    running: &[Running],
) -> IndexMap<String, HostState<'c>> {
    let mut states = IndexMap::new();
    for (name, r) in results {
        let Some(host) = cfg.hosts.get(name).filter(|_| r.reachable) else { continue };
        states.insert(
            name.clone(),
            HostState {
                host,
                free: r.gpus.iter().map(|g| (g.index, g.is_free())).collect(),
                cpu_free: host.cpu_slots,
                detected_mem_gb: r.gpus.iter().map(|g| g.mem_total_gb()).reduce(f64::min).unwrap_or(0.0),
                names: r.gpus.iter().map(|g| (g.index, g.name.clone())).collect(),
            },
        );
    }
    for row in running {
        let Some(st) = row.server.as_ref().and_then(|s| states.get_mut(s)) else { continue };
        if row.num_gpus == 0 {
            st.cpu_free -= 1;
        }
        for idx in row.gpu_indices.as_deref().unwrap_or("").split(',') {
            if py::isdigit(py::strip(idx)) {
                if let Some(free) = py::int(idx).and_then(|i| st.free.get_mut(&i)) {
                    *free = false;
                }
            }
        }
    }
    states
}

/// Greedily assign tasks (in queue order) to hosts. GPU tasks bin-pack onto hosts with the fewest
/// GPUs first, using only GPUs whose nvidia-smi name matches the task's --gpu-type; -G 0 tasks use
/// free CPU slots. `budget` caps the GPUs handed out in this round ([limits] max_gpus minus those
/// in use). Reserves what it assigns in `states`. Returns (task id, host, gpu indices).
pub fn assign_tasks(
    tasks: &[Task],
    states: &mut IndexMap<String, HostState>,
    cfg: &Config,
    mut budget: Option<i64>,
) -> Vec<(i64, String, Vec<i64>)> {
    let mut order: Vec<String> = states.keys().cloned().collect();
    order.sort_by_key(|n| states[n].free.len()); // stable: ties keep the poll order
    let mut assignments = vec![];
    for t in tasks {
        let n = t.num_gpus;
        if budget.is_some_and(|b| n > b) {
            continue;
        }
        let Ok(allowed) = cfg.select(t.host_spec.as_deref()) else { continue };
        let allowed: HashSet<&str> = allowed.iter().map(|h| h.name.as_str()).collect();
        for name in &order {
            if !allowed.contains(name.as_str()) {
                continue;
            }
            let st = states.get_mut(name).expect("order lists the states");
            if n == 0 {
                if st.cpu_free > 0 {
                    st.cpu_free -= 1;
                    assignments.push((t.id, name.clone(), vec![]));
                    break;
                }
                continue;
            }
            if st.mem_gb() < t.min_mem as f64 {
                continue;
            }
            let free: Vec<i64> = st
                .free
                .iter()
                .filter(|(i, ok)| **ok && gpu_type_matches(t.gpu_type.as_deref(), st.names.get(i).map_or("", |s| s)))
                .map(|(i, _)| *i)
                .collect();
            if free.len() as i64 >= n {
                let chosen: Vec<i64> = free[..n as usize].to_vec();
                for idx in &chosen {
                    st.free.insert(*idx, false); // reserve
                }
                assignments.push((t.id, name.clone(), chosen));
                if let Some(b) = budget.as_mut() {
                    *b -= n;
                }
                break;
            }
        }
    }
    assignments
}

// ── The daemon ────────────────────────────────────────────────────────────────

type Error = Box<dyn std::error::Error>;

/// (device, inode, modification time) of the running binary, to notice an upgrade.
fn exe_signature(path: &PathBuf) -> Option<(u64, u64, i64, i64)> {
    use std::os::unix::fs::MetadataExt;
    let m = std::fs::metadata(path).ok()?;
    Some((m.dev(), m.ino(), m.mtime(), m.mtime_nsec()))
}

pub struct Daemon {
    paths: Paths,
    con: Connection,
    transport: Transport,
    exe: Option<PathBuf>,
    exe_sig: Option<(u64, u64, i64, i64)>,
    last_snapshot: f64,
    queued_seen: HashSet<i64>,
    last_dispatch: f64,
    last_status: f64,
    dispatch_due: bool,
    status_due: bool,
    mode: Option<Mode>,
    unknown_hosts: HashSet<String>,
    _lock: File,
}

pub fn log(msg: &str) {
    let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    let mut out = std::io::stdout();
    let _ = writeln!(out, "[{ts}] {msg}");
    let _ = out.flush();
}

/// Take the daemon lock, or report the PID of the daemon that has it.
fn lock(paths: &Paths) -> Result<Result<File, i64>, Error> {
    use std::os::fd::AsRawFd;
    std::fs::create_dir_all(&paths.state)?;
    let mut f =
        std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(paths.lock_file())?;
    // SAFETY: flock on a file descriptor we own.
    if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        let holder = std::fs::read_to_string(paths.lock_file()).ok().and_then(|s| py::int(&s)).unwrap_or(0);
        return Ok(Err(holder));
    }
    f.set_len(0)?;
    write!(f, "{}", std::process::id())?;
    Ok(Ok(f))
}

impl Daemon {
    /// Start up: take the lock, open the database, load the config. Ok(None) if another daemon
    /// already runs (its PID is put back in daemon.pid).
    pub fn new(paths: Paths) -> Result<Option<Daemon>, Error> {
        let lock = match lock(&paths)? {
            Ok(f) => f,
            Err(holder) => {
                log(&format!("Another daemon (PID {holder}) is already running; this one exits."));
                if holder > 0 {
                    std::fs::write(paths.pid_file(), holder.to_string())?;
                }
                return Ok(None);
            }
        };
        std::fs::write(paths.pid_file(), std::process::id().to_string())?;
        let con = db::open(&paths)?;
        let cfg = load_config(&paths).map_err(|e| format!("task daemon: config error: {e}"))?;
        let exe = std::env::current_exe().ok();
        let exe_sig = exe.as_ref().and_then(exe_signature);
        Ok(Some(Daemon {
            transport: Transport::new(cfg, paths.clone()),
            paths,
            con,
            exe,
            exe_sig,
            last_snapshot: 0.0,
            queued_seen: HashSet::new(),
            last_dispatch: 0.0,
            last_status: 0.0,
            dispatch_due: true,
            status_due: true,
            mode: None,
            unknown_hosts: HashSet::new(),
            _lock: lock,
        }))
    }

    fn cfg(&self) -> &Config {
        &self.transport.cfg
    }

    pub fn run(&mut self) -> ! {
        log(&format!("Daemon started. Config: {}", self.cfg().source_label()));
        // Tasks started by the old (non-detached) version cannot be re-adopted.
        match self.con.execute(
            "UPDATE tasks SET status='lost', finished_at=? WHERE status='running' AND job_key IS NULL",
            [now()],
        ) {
            Ok(n) if n > 0 => log(&format!("Marked {n} task(s) from the previous version as 'lost'.")),
            Ok(_) => {}
            Err(e) => log(&format!("Error in daemon loop: {e}")),
        }
        if let Ok(n) =
            self.con.query_row("SELECT COUNT(*) FROM tasks WHERE status='running'", [], |r| r.get::<_, i64>(0))
        {
            if n > 0 {
                log(&format!("Re-adopting {n} running task(s)."));
            }
        }
        loop {
            let tick = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.tick()));
            let failed = match tick {
                Ok(Ok(())) => None,
                Ok(Err(e)) => Some(e.to_string()),
                Err(panic) => Some(
                    panic
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "panic".into()),
                ),
            };
            if let Some(e) = failed {
                log(&format!("Error in daemon loop: {e}"));
                std::thread::sleep(Duration::from_secs_f64(status_interval()));
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    }

    fn tick(&mut self) -> Result<(), Error> {
        self.maybe_restart_on_code_change();
        self.maybe_reload_config();
        let now = now();

        let mode = self.transport.mode(false);
        if Some(mode) != self.mode {
            log(&format!("Network mode: {}", self.transport.describe_mode(true)));
            self.mode = Some(mode);
        }

        if self.status_due || now - self.last_status >= status_interval() {
            self.status_due = false;
            self.last_status = now;
            self.poll_running()?;
        }

        let queued: HashSet<i64> = self
            .con
            .prepare("SELECT id FROM tasks WHERE status='queued'")?
            .query_map([], |r| r.get(0))?
            .collect::<Result<_, _>>()?;
        if !queued.is_subset(&self.queued_seen) {
            self.dispatch_due = true;
        }
        let any_queued = !queued.is_empty();
        self.queued_seen = queued;
        if any_queued && (self.dispatch_due || now - self.last_dispatch >= POLL_INTERVAL) {
            self.dispatch_due = false;
            self.last_dispatch = crate::transport::now();
            self.dispatch()?;
        }

        if self.snapshot_due() {
            let workers = self.cfg().workers();
            let results = query_hosts(&self.transport, &workers, 16, false);
            self.write_snapshot(&results);
        }
        Ok(())
    }

    // ── GPU snapshot for viewers (tui) ───────────────────────────────────────

    /// A viewer asked recently and the snapshot is older than its request or stale.
    fn snapshot_due(&self) -> bool {
        let requested = crate::config::mtime(&self.paths.snapshot_request());
        if requested == 0.0 {
            return false;
        }
        let now = now();
        if now - requested > SNAPSHOT_WANTED {
            return false;
        }
        requested > self.last_snapshot || now - self.last_snapshot >= POLL_INTERVAL
    }

    fn write_snapshot(&mut self, results: &IndexMap<String, ServerResult>) {
        self.last_snapshot = now();
        let keep: Vec<String> = self.cfg().workers().iter().map(|h| h.name.clone()).collect();
        if let Err(e) = write_snapshot(&self.paths.snapshot(), results, self.transport.mode(false), Some(&keep)) {
            log(&format!("Cannot write GPU snapshot: {e}"));
        }
    }

    /// Re-exec the binary after an update (install.sh replaces it; a rebuild rewrites it). Same
    /// PID, so the PID file stays valid; running jobs are re-adopted from the database like after
    /// any restart.
    fn maybe_restart_on_code_change(&self) {
        use std::os::unix::process::CommandExt;
        let Some(exe) = &self.exe else { return };
        let Some(sig) = exe_signature(exe) else { return }; // being replaced right now: next tick
        if Some(sig) == self.exe_sig {
            return;
        }
        log("Code changed on disk; restarting the daemon to pick it up.");
        let err = std::process::Command::new(exe).arg("__daemon").exec();
        log(&format!("Restart failed ({err}); carrying on with the running code."));
    }

    fn maybe_reload_config(&mut self) {
        let path = &self.paths.config;
        let mtime = if path.exists() { crate::config::mtime(path) } else { 0.0 };
        let current = if self.cfg().source.is_some() { self.cfg().mtime } else { 0.0 };
        if mtime == current {
            return;
        }
        match load_config(&self.paths) {
            Err(e) => {
                log(&format!("Config reload failed, keeping previous config: {e}"));
                self.transport.cfg.mtime = mtime; // don't retry until it changes again
            }
            Ok(cfg) => {
                let label = cfg.source_label();
                self.transport = Transport::new(cfg, self.paths.clone());
                self.unknown_hosts.clear();
                log(&format!("Config reloaded from {label}."));
            }
        }
    }

    // ── status / log sync ────────────────────────────────────────────────────

    fn tasks(&self, sql: &str) -> Result<Vec<Task>, Error> {
        Ok(self.con.prepare(sql)?.query_map([], Task::from_row)?.collect::<Result<_, _>>()?)
    }

    fn poll_running(&mut self) -> Result<(), Error> {
        let rows = self.tasks("SELECT * FROM tasks WHERE status='running' AND job_key IS NOT NULL")?;
        if rows.is_empty() {
            return Ok(());
        }
        let mut by_host: IndexMap<String, Vec<Task>> = IndexMap::new();
        for r in rows {
            let server = r.server.clone().unwrap_or_default();
            if !self.cfg().hosts.contains_key(&server) {
                if self.unknown_hosts.insert(server.clone()) {
                    log(&format!(
                        "Task {}: host '{}' is no longer configured; cannot poll it.",
                        r.id,
                        r.server.as_deref().unwrap_or("None")
                    ));
                }
                continue;
            }
            by_host.entry(server).or_default().push(r);
        }

        let hosts: Vec<&Host> = by_host.keys().map(|h| &self.cfg().hosts[h]).collect();
        let results = self.transport.fan_out(
            &hosts,
            |host| {
                let specs: Vec<(String, i64)> =
                    by_host[&host.name].iter().map(|r| (r.job_key.clone().unwrap_or_default(), r.log_offset)).collect();
                let out = self.transport.run(host, &jobs::status_script(&specs), STATUS_TIMEOUT);
                if out.rc == 0 {
                    jobs::parse_status(&out.stdout)
                } else {
                    None
                }
            },
            16,
        );
        let now = now();
        for (host_name, statuses) in results {
            let Some(statuses) = statuses else { continue }; // unreachable: leave tasks running, last_seen unchanged
            for row in &by_host[&host_name] {
                if let Some(st) = row.job_key.as_ref().and_then(|k| statuses.get(k)) {
                    self.apply_status(row, st, now)?;
                }
            }
        }
        Ok(())
    }

    fn apply_status(&mut self, row: &Task, st: &JobStatus, now: f64) -> Result<(), Error> {
        let (tid, key) = (row.id, row.job_key.clone().unwrap_or_default());
        let mut offset = row.log_offset;
        if !st.data.is_empty() {
            let appended = match &row.log_file {
                Some(path) => std::fs::OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(path)
                    .and_then(|mut f| f.write_all(&st.data))
                    .map_err(|e| e.to_string()),
                None => Err("no log file".to_string()),
            };
            match appended {
                Ok(()) => offset += st.data.len() as i64,
                Err(e) => log(&format!("Task {tid}: cannot append to log: {e}")),
            }
        }
        self.con.execute(
            "UPDATE tasks SET log_offset=?, last_seen=? WHERE id=? AND job_key=?",
            rusqlite::params![offset, now, tid, key],
        )?;

        if st.state == "running" {
            return Ok(());
        }
        if offset < st.size {
            self.status_due = true; // fetch the rest of the log right away
            return Ok(());
        }
        if st.state == "done" {
            self.finish(row, st.rc.unwrap_or(1))?;
        } else {
            self.update_running(tid, &key, "status='lost', exit_code=NULL, finished_at=?", vec![Sql::Real(now)])?;
            log(&format!(
                "Task {tid} lost on {} (job directory {}; not retrying).",
                row.server.as_deref().unwrap_or("None"),
                st.state
            ));
        }
        self.dispatch_due = true; // GPUs/slots were freed
        Ok(())
    }

    /// UPDATE a task only if it is still this running attempt.
    fn update_running(&self, tid: i64, key: &str, assignments: &str, mut params: Vec<Sql>) -> Result<bool, Error> {
        params.extend([Sql::Integer(tid), Sql::Text(key.to_string())]);
        let n = self.con.execute(
            &format!("UPDATE tasks SET {assignments} WHERE id=? AND job_key=? AND status='running'"),
            params_from_iter(params),
        )?;
        Ok(n > 0)
    }

    fn provenance(&self, event: &str, task_id: i64, row: Option<&Task>) {
        let fetched;
        let row = match row {
            Some(r) => r,
            None => match db::find_task(&self.con, task_id) {
                Ok(Some(r)) => {
                    fetched = r;
                    &fetched
                }
                _ => return,
            },
        };
        if provenance::notify(event, row, &self.cfg().taura, &[], &log) {
            log(&format!("Task {task_id}: recorded {event} as taura run {}.", provenance::run_id(row)));
        }
    }

    fn finish(&mut self, row: &Task, rc: i64) -> Result<(), Error> {
        let (tid, key) = (row.id, row.job_key.clone().unwrap_or_default());
        // Record this attempt as it ended, before a retry re-queues it (and bumps its attempt number).
        let ended = Task { exit_code: Some(rc), finished_at: Some(now()), ..row.clone() };
        self.provenance("finish", tid, Some(&ended));
        if rc == 0 {
            if self.update_running(tid, &key, "status='done', exit_code=0, finished_at=?", vec![Sql::Real(now())])? {
                log(&format!("Task {tid} finished — exit 0 (done)."));
            }
            return Ok(());
        }
        if row.retry_count < MAX_RETRIES {
            if self.update_running(tid, &key, &format!("{REQUEUE}, exit_code=?"), vec![Sql::Integer(rc)])? {
                log(&format!(
                    "Task {tid} failed (exit {rc}), re-queuing (attempt {}/{MAX_RETRIES}).",
                    row.retry_count + 1
                ));
            }
        } else if self.update_running(
            tid,
            &key,
            "status='failed', exit_code=?, finished_at=?",
            vec![Sql::Integer(rc), Sql::Real(now())],
        )? {
            log(&format!("Task {tid} failed (exit {rc}), no more retries ({MAX_RETRIES}/{MAX_RETRIES})."));
        }
        Ok(())
    }

    // ── dispatch ─────────────────────────────────────────────────────────────

    /// Query hosts and start queued tasks on free GPUs / CPU slots.
    fn dispatch(&mut self) -> Result<(), Error> {
        let rows = self.tasks("SELECT * FROM tasks WHERE status='queued' ORDER BY id")?;
        if rows.is_empty() {
            return Ok(());
        }
        let mut candidates: IndexMap<String, &Host> = IndexMap::new();
        for r in &rows {
            match self.cfg().select(r.host_spec.as_deref()) {
                Ok(hosts) => candidates.extend(hosts.into_iter().map(|h| (h.name.clone(), h))),
                Err(e) => log(&format!(
                    "Task {}: -H {}: {e}; it stays queued.",
                    r.id,
                    r.host_spec.as_deref().map_or("None".into(), py::repr_str)
                )),
            }
        }

        log(&format!("Polling {} host(s) for {} queued task(s)...", candidates.len(), rows.len()));
        let hosts: Vec<&Host> = candidates.values().copied().collect();
        let results = query_hosts(&self.transport, &hosts, 16, false);
        self.write_snapshot(&results);

        let running: Vec<Running> =
            self.tasks("SELECT * FROM tasks WHERE status='running'")?.iter().map(Running::from).collect();
        let cfg = self.cfg();
        let mut states = host_states(cfg, &results, &running);
        if states.is_empty() {
            let hint = results.values().any(|r| auth_hint(&r.error).is_some());
            log(&format!(
                "No reachable hosts (check SSH keys/agent{}).",
                if hint { "; run `task connect`" } else { "" }
            ));
            return Ok(());
        }

        let mut names: Vec<&String> = states.keys().collect();
        names.sort();
        let avail: Vec<String> = names
            .iter()
            .map(|n| {
                let s = &states[*n];
                let free = s.free.values().filter(|f| **f).count();
                format!("{n}:{free}{}", if s.cpu_free > 0 { format!("+{}cpu", s.cpu_free) } else { String::new() })
            })
            .collect();
        let budget = gpu_budget(cfg, &running);
        let cap = match (budget, cfg.max_gpus) {
            (Some(b), Some(m)) => format!("  (max_gpus {m}: {} in use)", m - b),
            _ => String::new(),
        };
        log(&format!("Free: {}{cap}", avail.join(", ")));

        let assignments = assign_tasks(&rows, &mut states, cfg, budget);
        if assignments.is_empty() {
            log(&format!(
                "No free slots available{}; tasks remain queued.",
                if budget == Some(0) { " within max_gpus" } else { "" }
            ));
            return Ok(());
        }
        let assignments: Vec<(i64, Host, Vec<i64>)> =
            assignments.into_iter().map(|(t, h, g)| (t, cfg.hosts[&h].clone(), g)).collect();
        for (task_id, host, gpu_indices) in assignments {
            self.launch(task_id, &host, &gpu_indices)?;
        }
        Ok(())
    }

    fn launch(&mut self, task_id: i64, host: &Host, gpu_indices: &[i64]) -> Result<(), Error> {
        let Some(row) = db::find_task(&self.con, task_id)? else { return Ok(()) };
        let gpu_str = gpu_indices.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",");
        let log_path = self.paths.logs().join(format!("{task_id}.log"));
        let key = jobs::new_job_key(task_id);
        let wdir = row.working_dir.clone();
        let rdir = self.cfg().remote_dir(host, &wdir, row.repo_root.as_deref());

        // Reserve in DB before launching, so a daemon restart mid-launch can find (or give up on)
        // the job directory.
        let n = self.con.execute(
            "UPDATE tasks SET status='running', server=?, gpu_indices=?, started_at=?, log_file=?, job_key=?, \
             remote_pid=NULL, log_offset=0, remote_dir=?, last_seen=NULL, finished_at=NULL WHERE id=? AND status='queued'",
            rusqlite::params![host.name, gpu_str, now(), log_path.display().to_string(), key, rdir, task_id],
        )?;
        if n == 0 {
            return Ok(()); // cancelled in the meantime
        }

        // A header for the log
        let mapped = if rdir != wdir { format!("  (mapped from {wdir})") } else { String::new() };
        let header = format!(
            "# task {task_id}: {}\n# server: {}  CUDA_VISIBLE_DEVICES: {}\n# started: {}\n# working_dir: {rdir}{mapped}\n{}\n",
            row.command,
            host.name,
            if gpu_str.is_empty() { "(none)" } else { &gpu_str },
            isoformat_now(),
            "─".repeat(60)
        );
        std::fs::write(&log_path, header)?;

        // Multi-GPU jobs sleep briefly before starting to reduce the chance of torchrun port
        // collisions when several launch at the same time.
        let probe = provenance::enabled(&row, &self.cfg().taura).then(|| provenance::probe_command(&row.command));
        let script = jobs::launch_script(&key, &rdir, &gpu_str, &row.command, gpu_indices.len() > 1, probe.as_deref());
        let out = self.transport.run(host, &script, LAUNCH_TIMEOUT);
        if let Some(pid) = jobs::parse_launch(&out.stdout) {
            let git = jobs::parse_git(&out.stdout);
            let dirty = git.as_ref().map(|_| jobs::parse_dirty(&out.stdout).join("\n"));
            self.con.execute(
                "UPDATE tasks SET remote_pid=?, git_commit=?, git_dirty=?, git_dirty_paths=? WHERE id=? AND job_key=?",
                rusqlite::params![
                    pid,
                    git.as_ref().map(|g| g.0.clone()),
                    git.as_ref().map(|g| g.1),
                    dirty,
                    task_id,
                    key
                ],
            )?;
            log(&format!(
                "Task {task_id} dispatched → {} [CUDA_VISIBLE_DEVICES={gpu_str}]  (job {key}, PGID {pid})",
                host.name
            ));
            self.provenance("start", task_id, None);
            return Ok(());
        }
        if out.rc == 124 {
            // The job may or may not have started; the status poll will tell.
            log(&format!("Task {task_id}: launch on {} timed out; will check its job directory.", host.name));
            return Ok(());
        }

        let err = py::strip(&out.stderr);
        let reason = py::splitlines(err)
            .last()
            .filter(|_| !err.is_empty())
            .map_or(format!("exit {}", out.rc), |l| l.to_string());
        let hint = auth_hint(&out.stderr);
        let detail = if err.is_empty() { py::strip(&out.stdout) } else { err };
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&log_path) {
            let _ = writeln!(f, "gpu-spooler: launch on {} failed: {detail}", host.name);
        }
        if row.retry_count < MAX_RETRIES {
            self.update_running(task_id, &key, REQUEUE, vec![])?;
            log(&format!(
                "Task {task_id} failed to launch on {}: {reason}{} — re-queuing (attempt {}/{MAX_RETRIES}).",
                host.name,
                hint.map_or(String::new(), |h| format!(" ({h})")),
                row.retry_count + 1
            ));
        } else {
            self.update_running(task_id, &key, "status='failed', finished_at=?", vec![Sql::Real(now())])?;
            log(&format!(
                "Task {task_id} failed to launch on {}: {reason} — no more retries ({MAX_RETRIES}/{MAX_RETRIES}).",
                host.name
            ));
        }
        Ok(())
    }
}

/// `datetime.now().isoformat()`: local time, microseconds unless they are zero.
fn isoformat_now() -> String {
    let now = chrono::Local::now();
    if now.timestamp_subsec_micros() == 0 {
        now.format("%Y-%m-%dT%H:%M:%S").to_string()
    } else {
        now.format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
    }
}

/// `task __daemon`: run until killed. Returns the exit code if it can't start.
pub fn main(paths: Paths) -> i32 {
    match Daemon::new(paths) {
        Ok(Some(mut d)) => d.run(),
        Ok(None) => 0,
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuInfo;

    fn gpu(index: i64, name: &str, used: i64, total: i64) -> GpuInfo {
        GpuInfo {
            index,
            name: name.into(),
            mem_used_mb: used,
            mem_free_mb: total - used,
            mem_total_mb: total,
            utilization_pct: 0,
            processes: vec![],
        }
    }

    fn task(id: i64, n: i64) -> Task {
        Task { id, num_gpus: n, status: "queued".into(), ..Default::default() }
    }

    #[test]
    fn detected_memory_wins_over_a_wrong_label() {
        let paths = Paths::for_home("/h");
        let cfg = crate::config::parse_config(
            &"[hosts.big]\ngpu_mem_gb = 80\n[hosts.small]\ngpu_mem_gb = 10\n".parse().unwrap(),
            &paths,
        )
        .unwrap();
        let results: IndexMap<String, ServerResult> = [
            ("big".to_string(), ServerResult::up("big", vec![gpu(0, "NVIDIA RTX A5000", 1, 24564)])),
            ("small".to_string(), ServerResult::up("small", vec![])),
        ]
        .into();
        let mut states = host_states(&cfg, &results, &[]);
        assert!((states["big"].mem_gb() - 25.76).abs() < 0.01); // detected, not the label's 80
        assert_eq!(states["small"].mem_gb(), 10.0); // nothing detected: the configured value
        let t = Task { min_mem: 40, ..task(1, 1) };
        assert!(assign_tasks(&[t], &mut states, &cfg, None).is_empty());
    }
}
