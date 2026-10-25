//! The tui's state and what each key does. Drawing is in view.rs; both work without a terminal,
//! so the tests drive [`App::key`] and render to a TestBackend.

use crate::ctx::Ctx;
use gpu_spooler::core::{self, SubmitOpts};
use gpu_spooler::db::{self, Task};
use gpu_spooler::fmt::{one_line, truncate};
use gpu_spooler::paths::Paths;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub const TASK_REFRESH: Duration = Duration::from_millis(1000);
pub const LOG_REFRESH: Duration = Duration::from_millis(500);
/// Re-read gpu_snapshot.json.
pub const SNAPSHOT_REFRESH: Duration = Duration::from_secs(2);
/// Ask the daemon to keep the snapshot fresh.
pub const SNAPSHOT_PING: Duration = Duration::from_secs(30);
const LOG_TAIL: u64 = 256 * 1024;
const LOG_LINES: usize = 20000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Log,
    Gpus,
    Info,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info,
    Warning,
    Error,
}

#[derive(Clone, Debug)]
pub struct Note {
    pub text: String,
    pub level: Level,
    pub until: Instant,
}

/// What a confirmed question does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Confirmed {
    Kill(i64),
    Clear,
}

/// The new-task form: the same fields as `task submit -g/-m/-H/-n/-C CMD`.
#[derive(Clone, Debug)]
pub struct Form {
    pub fields: Vec<(&'static str, String, &'static str)>, // label, value, placeholder
    pub focus: usize,
}

impl Form {
    pub fn new(cwd: String) -> Form {
        Form {
            fields: vec![
                ("Command", String::new(), "python3 train.py --epochs 50"),
                ("GPUs (-G)", "1".into(), ""),
                ("Min VRAM GB (-m)", "0".into(), ""),
                ("Hosts (-H)", String::new(), "all workers, or e.g. cuda14,local"),
                ("Label (-n)", String::new(), ""),
                ("Directory (-d)", cwd, ""),
            ],
            focus: 0,
        }
    }

    fn value(&self, i: usize) -> String {
        gpu_spooler::py::strip(&self.fields[i].1).to_string()
    }

    /// The submit options, or why the form can't be submitted.
    pub fn opts(&self) -> Result<SubmitOpts, String> {
        let command = self.value(0);
        if command.is_empty() {
            return Err(String::new()); // the bell: nothing to queue
        }
        let int = |i: usize| {
            let v = self.value(i);
            if v.is_empty() {
                Ok(0)
            } else {
                gpu_spooler::py::int(&v).ok_or(format!("{}: not a number", self.fields[i].0))
            }
        };
        let opt = |i: usize| Some(self.value(i)).filter(|v| !v.is_empty());
        Ok(SubmitOpts {
            command,
            num_gpus: int(1)?,
            min_mem: int(2)?,
            host_spec: opt(3),
            name: opt(4),
            working_dir: opt(5),
            start: true,
            check_repo: true,
            ..Default::default()
        })
    }
}

#[derive(Clone, Debug)]
pub enum Modal {
    Confirm { question: String, then: Confirmed },
    Submit(Form),
}

/// Work done off the UI thread (it talks to servers), reported back as a note.
enum Job {
    Kill(i64),
    Resubmit(i64, bool),
    Submit(Box<SubmitOpts>),
    Sync,
    Clear,
}

pub struct App {
    pub paths: Paths,
    ctx: Ctx,
    /// The directory tui was started in (`s` syncs its repo, the form's default directory).
    pub cwd: String,
    pub rows: Vec<Task>,
    pub selected: usize,
    pub tab: Tab,
    pub follow: bool,
    pub modal: Option<Modal>,
    pub notes: Vec<Note>,
    pub subtitle: String,
    pub details: Vec<(String, String)>,
    pub gpu_map: Vec<(String, ratatui::style::Style)>,
    /// The selected task's log, and how far it has been read (None: no log file yet).
    pub log: String,
    log_task: Option<i64>,
    log_offset: Option<u64>,
    /// Lines up from the end of the log (0: following the end).
    pub log_scroll: usize,
    pub quit: bool,
    pub bell: bool,
    /// Start the daemon when submitting, as `task submit` does (off in tests).
    pub start_daemon: bool,
    done_tx: mpsc::Sender<Note>,
    done_rx: mpsc::Receiver<Note>,
    last: [Option<Instant>; 4],
}

fn note(text: impl Into<String>, level: Level, secs: u64) -> Note {
    Note { text: text.into(), level, until: Instant::now() + Duration::from_secs(secs) }
}

impl App {
    pub fn new(paths: Paths, cwd: String) -> App {
        let (done_tx, done_rx) = mpsc::channel();
        App {
            ctx: Ctx::new(paths.clone()),
            paths,
            cwd,
            rows: vec![],
            selected: 0,
            tab: Tab::Log,
            follow: false,
            modal: None,
            notes: vec![],
            subtitle: String::new(),
            details: vec![],
            gpu_map: vec![],
            log: String::new(),
            log_task: None,
            log_offset: None,
            log_scroll: 0,
            quit: false,
            bell: false,
            start_daemon: true,
            done_tx,
            done_rx,
            last: [None; 4],
        }
    }

    pub fn selected_task(&self) -> Option<&Task> {
        self.rows.get(self.selected)
    }

    // ── data refresh ─────────────────────────────────────────────────────────

    /// Refresh whatever is due (call often); `force` refreshes everything now.
    pub fn tick(&mut self, force: bool) {
        let now = Instant::now();
        let due = |last: &mut Option<Instant>, every: Duration| {
            let due = force || last.is_none_or(|l| now.duration_since(l) >= every);
            if due {
                *last = Some(now);
            }
            due
        };
        if due(&mut self.last[3], SNAPSHOT_PING) {
            let _ = core::request_gpu_snapshot(&self.paths);
        }
        if due(&mut self.last[0], TASK_REFRESH) {
            self.refresh_tasks();
        }
        if due(&mut self.last[1], LOG_REFRESH) {
            self.refresh_log();
        }
        if due(&mut self.last[2], SNAPSHOT_REFRESH) {
            self.refresh_gpus();
        }
        while let Ok(n) = self.done_rx.try_recv() {
            self.notes.push(n);
        }
        self.notes.retain(|n| n.until > now);
    }

    pub fn refresh_tasks(&mut self) {
        let keep = self.selected_task().map(|t| t.id);
        let Ok(con) = db::open(&self.paths) else { return };
        self.rows = db::list_tasks(&con, Some(200)).unwrap_or_default();
        self.selected = keep.and_then(|id| self.rows.iter().position(|t| t.id == id)).unwrap_or(0);
        let counts = db::task_counts(&con).unwrap_or_default();
        let daemon = if core::daemon_pid(&self.paths).is_some() { "daemon running" } else { "DAEMON STOPPED (d)" };
        let net = self.ctx.transport().map_or("?".to_string(), |t| t.describe_mode(false));
        let count = |k: &str| counts.get(k).copied().unwrap_or(0);
        self.subtitle = format!("{} running · {} queued · {daemon} · network {net}", count("running"), count("queued"));
        self.refresh_details();
    }

    pub fn refresh_details(&mut self) {
        let snapshot = core::read_gpu_snapshot(&self.paths);
        let cfg = self.ctx.cfg().ok();
        self.details = match self.selected_task() {
            None => vec![],
            Some(t) => core::task_details(t, cfg, snapshot.as_ref(), gpu_spooler::transport::now()),
        };
    }

    fn append_log(&mut self, text: &str) {
        self.log.push_str(text);
        let lines = self.log.matches('\n').count();
        if lines > LOG_LINES {
            let cut = self.log.match_indices('\n').nth(lines - LOG_LINES - 1).map_or(0, |(i, _)| i + 1);
            self.log.drain(..cut);
        }
    }

    pub fn refresh_log(&mut self) {
        let task = self.selected_task().cloned();
        let tid = task.as_ref().map(|t| t.id);
        let path = task.as_ref().and_then(|t| t.log_file.clone()).filter(|l| !l.is_empty()).map(PathBuf::from);
        let exists = path.as_ref().is_some_and(|p| p.exists());
        if tid != self.log_task {
            self.log_task = tid;
            self.log.clear();
            self.log_scroll = 0;
            if !exists {
                self.log = if task.is_some() { "(no output yet)\n".into() } else { String::new() };
                self.log_offset = None;
                return;
            }
            if let Ok((text, size)) = core::read_log_tail(path.as_ref().unwrap(), LOG_TAIL) {
                self.append_log(&text);
                self.log_offset = Some(size);
            }
            return;
        }
        let Some(path) = path.filter(|_| exists) else { return };
        let size = std::fs::metadata(&path).map_or(0, |m| m.len());
        let offset = match self.log_offset {
            None => {
                self.log.clear(); // the log appeared since we looked
                0
            }
            Some(o) if size < o => {
                self.log.clear(); // rewritten (a retry)
                0
            }
            Some(o) => o,
        };
        if let Ok((text, new)) = core::read_log_from(&path, offset) {
            self.append_log(&text);
            self.log_offset = Some(new);
        }
    }

    pub fn refresh_gpus(&mut self) {
        let running: Vec<Task> = self.rows.iter().filter(|t| t.status == "running").cloned().collect();
        let workers: Vec<String> =
            self.ctx.cfg().map(|c| c.workers().iter().map(|h| h.name.clone()).collect()).unwrap_or_default();
        let snapshot = core::read_gpu_snapshot(&self.paths);
        let daemon = core::daemon_pid(&self.paths).is_some();
        self.gpu_map =
            super::model::gpu_map(snapshot.as_ref(), &running, &workers, gpu_spooler::transport::now(), daemon);
    }

    /// The selection moved: show that task's log and details.
    fn selection_changed(&mut self) {
        self.refresh_log();
        self.refresh_details();
    }

    // ── keys ─────────────────────────────────────────────────────────────────

    pub fn key(&mut self, k: KeyEvent) {
        if k.modifiers.contains(KeyModifiers::CONTROL) && k.code == KeyCode::Char('c') {
            self.quit = true;
            return;
        }
        match self.modal.take() {
            Some(Modal::Confirm { question, then }) => match k.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => self.confirmed(then),
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {}
                _ => self.modal = Some(Modal::Confirm { question, then }),
            },
            Some(Modal::Submit(form)) => self.form_key(form, k),
            None => self.browse_key(k),
        }
    }

    fn form_key(&mut self, mut form: Form, k: KeyEvent) {
        let n = form.fields.len();
        match k.code {
            KeyCode::Esc => return,
            KeyCode::Enter => match form.opts() {
                Ok(opts) => return self.spawn(Job::Submit(Box::new(SubmitOpts { start: self.start_daemon, ..opts }))),
                Err(e) if e.is_empty() => self.bell = true,
                Err(e) => self.notes.push(note(e, Level::Error, 8)),
            },
            KeyCode::Tab | KeyCode::Down => form.focus = (form.focus + 1) % n,
            KeyCode::BackTab | KeyCode::Up => form.focus = (form.focus + n - 1) % n,
            KeyCode::Backspace => {
                form.fields[form.focus].1.pop();
            }
            KeyCode::Char(c) if !k.modifiers.contains(KeyModifiers::CONTROL) => form.fields[form.focus].1.push(c),
            _ => {}
        }
        self.modal = Some(Modal::Submit(form));
    }

    fn move_to(&mut self, i: usize) {
        let i = i.min(self.rows.len().saturating_sub(1));
        if i != self.selected {
            self.selected = i;
            self.selection_changed();
        }
    }

    fn browse_key(&mut self, k: KeyEvent) {
        if self.follow {
            match k.code {
                KeyCode::Up => return self.log_scroll += 1,
                KeyCode::Down => return self.log_scroll = self.log_scroll.saturating_sub(1),
                KeyCode::PageUp => return self.log_scroll += 20,
                KeyCode::PageDown => return self.log_scroll = self.log_scroll.saturating_sub(20),
                KeyCode::End => return self.log_scroll = 0,
                _ => {}
            }
        }
        match k.code {
            KeyCode::Up => self.move_to(self.selected.saturating_sub(1)),
            KeyCode::Down => self.move_to(self.selected + 1),
            KeyCode::PageUp => self.move_to(self.selected.saturating_sub(10)),
            KeyCode::PageDown => self.move_to(self.selected + 10),
            KeyCode::Home => self.move_to(0),
            KeyCode::End => self.move_to(usize::MAX),
            KeyCode::Char('q') => self.quit = true,
            KeyCode::Char('l') => self.tab = Tab::Log,
            KeyCode::Char('g') => {
                self.tab = Tab::Gpus;
                let _ = core::request_gpu_snapshot(&self.paths);
            }
            KeyCode::Char('i') => self.tab = Tab::Info,
            KeyCode::Char('f') => {
                self.tab = Tab::Log;
                self.follow = !self.follow;
                self.log_scroll = 0;
            }
            KeyCode::Char('k') => self.ask_kill(),
            KeyCode::Char('r') => {
                if let Some(id) = self.selected_task().map(|t| t.id) {
                    self.spawn(Job::Resubmit(id, self.start_daemon));
                }
            }
            KeyCode::Char('n') => self.modal = Some(Modal::Submit(Form::new(self.cwd.clone()))),
            KeyCode::Char('s') => {
                self.notes.push(note("Syncing git repo...", Level::Info, 5));
                self.spawn(Job::Sync);
            }
            KeyCode::Char('c') => {
                self.modal = Some(Modal::Confirm {
                    question: "Remove all finished tasks, their logs and job directories?".into(),
                    then: Confirmed::Clear,
                })
            }
            KeyCode::Char('d') => {
                let msg = match core::start_daemon(&self.paths) {
                    Ok(Some(pid)) => format!("Daemon started (PID {pid})."),
                    Ok(None) => "Daemon is already running.".into(),
                    Err(e) => format!("Cannot start the daemon: {e}"),
                };
                self.notes.push(note(msg, Level::Info, 5));
                let _ = core::request_gpu_snapshot(&self.paths);
            }
            _ => {}
        }
    }

    fn ask_kill(&mut self) {
        let Some(t) = self.selected_task().filter(|t| core::ACTIVE.contains(&t.status.as_str())) else {
            self.notes.push(note("Select a queued or running task.", Level::Warning, 5));
            return;
        };
        let verb = if t.status == "queued" { "Cancel" } else { "Kill" };
        let name = t.name.as_deref().filter(|n| !n.is_empty()).unwrap_or(&t.command);
        let question = format!("{verb} task {}: {}?", t.id, truncate(&one_line(name), 50));
        self.modal = Some(Modal::Confirm { question, then: Confirmed::Kill(t.id) });
    }

    fn confirmed(&mut self, then: Confirmed) {
        match then {
            Confirmed::Kill(id) => self.spawn(Job::Kill(id)),
            Confirmed::Clear => self.spawn(Job::Clear),
        }
    }

    // ── background work ──────────────────────────────────────────────────────

    fn spawn(&mut self, job: Job) {
        let (paths, cwd, tx) = (self.paths.clone(), self.cwd.clone(), self.done_tx.clone());
        std::thread::spawn(move || {
            let _ = tx.send(run_job(job, paths, &cwd));
        });
    }

    /// Wait for background work (tests): up to `timeout` for the next note.
    #[cfg(test)]
    pub fn wait_for_note(&mut self, timeout: Duration) -> Option<Note> {
        let n = self.done_rx.recv_timeout(timeout).ok()?;
        self.notes.push(n.clone());
        Some(n)
    }
}

fn run_job(job: Job, paths: Paths, cwd: &str) -> Note {
    let ctx = Ctx::new(paths);
    let transport = match ctx.transport() {
        Ok(t) => t,
        Err(e) => return note(e.message, Level::Error, 10),
    };
    let submitted = |r: Result<core::Submitted, core::TaskError>| match r {
        Err(e) => note(e.to_string(), Level::Error, 5),
        Ok(sub) => {
            let mut msg = format!("Queued task {}: {}", sub.task_id, truncate(&sub.command, 60));
            for w in &sub.warnings {
                msg += &format!("\nWarning: {w}");
            }
            note(msg, if sub.warnings.is_empty() { Level::Info } else { Level::Warning }, 5)
        }
    };
    match job {
        Job::Kill(id) => match core::kill(transport, id) {
            Ok(r) if r.confirmed => note(r.message, Level::Info, 8),
            Ok(r) => note(r.message, Level::Warning, 20),
            Err(e) => note(e.to_string(), Level::Error, 8),
        },
        Job::Resubmit(id, start) => submitted(core::resubmit(transport, id, start)),
        Job::Submit(opts) => submitted(core::submit(transport, &opts)),
        Job::Clear => match core::clear_finished(transport) {
            Ok(res) => {
                let mut msg = format!("Removed {} finished task(s).", res.tasks);
                if !res.failed_hosts.is_empty() {
                    msg += &format!(" Could not clean job directories on: {}", res.failed_hosts.join(", "));
                }
                note(msg, Level::Info, 5)
            }
            Err(e) => note(e.to_string(), Level::Error, 8),
        },
        Job::Sync => {
            let info = match core::repo_for(transport, Some(cwd), None) {
                Ok(i) => i,
                Err(e) => return note(e.to_string(), Level::Error, 10),
            };
            let results = core::sync_repo(transport, &info);
            let mut lines: Vec<String> = results
                .iter()
                .map(|r| format!("{} {}: {}", if r.ok { "OK" } else { "FAILED" }, r.target.label(), r.message))
                .collect();
            let paths: Vec<String> = info.copies.iter().map(|t| t.path.clone()).collect();
            let busy = db::open(&transport.paths).ok().and_then(|con| core::running_in(&con, &paths).ok()).unwrap_or(0);
            if busy > 0 {
                lines.push(format!("{busy} running task(s) use this checkout."));
            }
            let ok = results.iter().all(|r| r.ok);
            note(lines.join("\n"), if ok { Level::Info } else { Level::Error }, 12)
        }
    }
}
